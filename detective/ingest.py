"""Load a case folder into structured documents and citable passages."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Case, Document, Passage

LINE_ORIENTED_TYPES = {"system_log", "cctv", "chat", "phone_record"}
_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n(.*)\Z", re.DOTALL)


def parse_document(raw: str, source: str = "<memory>") -> Document:
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError(f"{source}: missing front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    missing = {"id", "type", "title", "date", "event_date"} - meta.keys()
    if missing:
        raise ValueError(f"{source}: front matter is missing {sorted(missing)}")
    return Document(text=match.group(2).strip(), **meta)


def load_case(path: str | Path) -> Case:
    """Read case.json and its documents. The answer key stays out of the investigation."""
    root = Path(path)
    case = Case.model_validate(json.loads((root / "case.json").read_text(encoding="utf-8")))
    for file in sorted((root / "documents").glob("*.txt"), key=_natural_key):
        case.documents.append(parse_document(file.read_text(encoding="utf-8"), str(file)))
    for lead in case.leads:
        file = root / lead.document
        case.sealed[lead.id] = parse_document(file.read_text(encoding="utf-8"), str(file))
    return case


def split_passages(doc: Document) -> list[Passage]:
    """Split on lines for log-like documents and on blank lines for prose."""
    if doc.type in LINE_ORIENTED_TYPES or " | " in doc.text or _looks_like_list(doc.text):
        chunks = doc.text.splitlines()
    else:
        chunks = re.split(r"\n\s*\n", doc.text)
    chunks = [c.strip() for c in chunks if c.strip()]
    return [Passage(id=f"{doc.id}:p{i}", doc_id=doc.id, index=i, text=c) for i, c in enumerate(chunks, 1)]


def _looks_like_list(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 3 and "\n\n" not in text.replace("\r", "")


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]
