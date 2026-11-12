"""Create cases from the web form or from a zip archive, and validate them.

Everything here handles user-supplied input, so paths are never taken from a
request without checking, and a case is only saved after it loads and validates.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from .ingest import load_case
from .models import DOC_TIERS, Case

MAX_ARCHIVE_BYTES = 10_000_000
MAX_EXPANDED_BYTES = 50_000_000
MAX_FILES = 500
ALLOWED_SUFFIXES = {".txt", ".json"}


class CaseError(ValueError):
    """The case cannot be used; the message is safe to show to the user."""


class DocumentDraft(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    type: str
    date: str
    speaker: str | None = None
    device: str | None = None
    text: str = Field(min_length=1)


class PersonDraft(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    role: str = ""
    aliases: list[str] = Field(default_factory=list)


class LocationDraft(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    within: str | None = None
    aliases: list[str] = Field(default_factory=list)


class CaseDraft(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    briefing: str = Field(min_length=1)
    crime_summary: str = Field(min_length=1)
    scene: str = Field(min_length=1)
    window_start: str
    window_end: str
    suspects: list[PersonDraft]
    people: list[PersonDraft] = Field(default_factory=list)
    locations: list[LocationDraft]
    documents: list[DocumentDraft]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50] or "case"


def validate_case(case: Case) -> None:
    if not case.suspects:
        raise CaseError("A case needs at least one suspect.")
    if not case.documents:
        raise CaseError("A case needs at least one document.")
    names = [loc.name for loc in case.locations]
    if case.crime.location not in names:
        raise CaseError(f"The crime scene '{case.crime.location}' must be one of the case's locations.")
    for location in case.locations:
        if location.within and location.within not in names:
            raise CaseError(f"'{location.name}' is inside '{location.within}', which is not a listed location.")
    for label, value in (("start", case.crime.window_start), ("end", case.crime.window_end)):
        _parse_time(value, f"crime window {label}")
    if case.crime.window_start >= case.crime.window_end:
        raise CaseError("The crime window must end after it starts.")
    identifiers = [d.id for d in case.documents] + [d.id for d in case.sealed.values()]
    duplicates = {i for i in identifiers if identifiers.count(i) > 1}
    if duplicates:
        raise CaseError(f"Document IDs must be unique: {sorted(duplicates)}")
    people = [p.name for p in case.everyone()]
    if len(people) != len(set(people)):
        raise CaseError("Each person may only be listed once.")


def _parse_time(value: str, label: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except ValueError as error:
        raise CaseError(f"The {label} must look like 2025-03-13T18:50.") from error


def unique_directory(root: Path, base: str) -> tuple[str, Path]:
    slug = base
    counter = 2
    while (root / slug).exists():
        slug = f"{base}_{counter}"
        counter += 1
    return slug, root / slug


def create_from_draft(draft: CaseDraft, root: Path, reserved: set[str]) -> str:
    event_date = draft.window_start[:10]
    manifest = {
        "id": "",
        "title": draft.title,
        "difficulty": "custom",
        "briefing": draft.briefing,
        "crime": {
            "summary": draft.crime_summary,
            "location": draft.scene,
            "window_start": draft.window_start,
            "window_end": draft.window_end,
            "artifact_ids": [],
        },
        "suspects": [p.model_dump() for p in draft.suspects],
        "people": [p.model_dump() for p in draft.people],
        "locations": [loc.model_dump(exclude_none=True) for loc in draft.locations],
        "leads": [],
    }
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp) / "case"
        (folder / "documents").mkdir(parents=True)
        for index, document in enumerate(draft.documents, 1):
            if document.type not in DOC_TIERS:
                raise CaseError(f"Unknown document type: {document.type}")
            lines = [
                "---",
                f"id: C{index}",
                f"type: {document.type}",
                f"title: {_one_line(document.title)}",
                f"date: {_one_line(document.date)}",
                f"event_date: {event_date}",
            ]
            if document.speaker:
                lines.append(f"speaker: {_one_line(document.speaker)}")
            if document.device:
                lines.append(f"device: {_one_line(document.device)}")
            lines += ["---", document.text.strip(), ""]
            (folder / "documents" / f"C{index}.txt").write_text("\n".join(lines), encoding="utf-8")
        return _install(manifest, folder, root, reserved, slugify(draft.title))


def import_archive(data: bytes, root: Path, reserved: set[str]) -> str:
    if len(data) > MAX_ARCHIVE_BYTES:
        raise CaseError("The archive is larger than 10 MB.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise CaseError("That file is not a valid zip archive.") from error
    with archive, tempfile.TemporaryDirectory() as temp:
        folder = Path(temp) / "case"
        _extract_safely(archive, folder)
        base = _find_case_root(folder)
        try:
            manifest = json.loads((base / "case.json").read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise CaseError("case.json is not valid JSON.") from error
        slug = slugify(str(manifest.get("id") or manifest.get("title") or "imported_case"))
        return _install(manifest, base, root, reserved, slug)


def _extract_safely(archive: zipfile.ZipFile, target: Path) -> None:
    members = [m for m in archive.infolist() if not m.is_dir()]
    if len(members) > MAX_FILES:
        raise CaseError(f"The archive has more than {MAX_FILES} files.")
    if sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
        raise CaseError("The archive expands to more than 50 MB.")
    for member in members:
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in member.filename:
            raise CaseError(f"Unsafe path in archive: {member.filename}")
        if path.name.startswith(".") or "__MACOSX" in path.parts:
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise CaseError(f"Only .txt and .json files are allowed: {member.filename}")
        destination = target.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))


def _find_case_root(folder: Path) -> Path:
    candidates = list(folder.glob("case.json")) + list(folder.glob("*/case.json"))
    if len(candidates) != 1:
        raise CaseError("The archive needs exactly one case.json, at the top level or in one folder.")
    return candidates[0].parent


def _install(manifest: dict, folder: Path, root: Path, reserved: set[str], slug: str) -> str:
    """Validate in a temporary folder, then move the case into place."""
    root.mkdir(parents=True, exist_ok=True)
    while slug in reserved:
        slug = f"{slug}_copy"
    case_id, destination = unique_directory(root, slug)
    manifest["id"] = case_id
    (folder / "case.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _check_lead_paths(manifest, folder)
    try:
        case = load_case(folder)
    except CaseError:
        raise
    except (ValueError, KeyError, OSError) as error:
        raise CaseError(f"The case could not be loaded: {_short(error)}") from error
    validate_case(case)
    shutil.copytree(folder, destination)
    return case_id


def _check_lead_paths(manifest: dict, folder: Path) -> None:
    for lead in manifest.get("leads", []):
        path = PurePosixPath(str(lead.get("document", "")).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise CaseError(f"Unsafe lead path: {lead.get('document')}")
        if not folder.joinpath(*path.parts).is_file():
            raise CaseError(f"Lead {lead.get('id')} points to a missing document: {lead.get('document')}")


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _short(error: Exception) -> str:
    text = str(error).splitlines()[0] if str(error) else type(error).__name__
    return text[:200]
