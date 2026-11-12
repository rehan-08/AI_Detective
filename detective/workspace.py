"""Application service: one workspace per case, with background runs and saved state.

The web layer stays thin; this module owns what a run is, when it may start, and how
its result survives a restart.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .authoring import CaseDraft, create_from_draft, import_archive
from .evaluation import has_solution, load_solution, score
from .heuristic import HeuristicExtractor
from .ingest import load_case, split_passages
from .investigation import Investigation
from .models import Answer, Case, Document
from .report import render_report
from .rules import RuleReasoner
from .store import EvidenceStore

logger = logging.getLogger(__name__)
BUILTIN_CASES = Path(__file__).resolve().parent.parent / "cases"
OFFLINE, CLAUDE, GROQ = "offline", "claude", "groq"


class Conflict(Exception):
    """The request cannot run now, for example because a run is in progress."""


class NotFound(Exception):
    pass


NO_KEY_MESSAGE = "Claude is not configured. Set ANTHROPIC_API_KEY, or run 'ant auth login', and start again."
NO_GROQ_KEY_MESSAGE = "Groq is not configured. Set GROQ_API_KEY and start again."


def claude_configured() -> bool:
    """True for an API key, an auth token or a profile saved by 'ant auth login'."""
    variables = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
    return any(os.environ.get(name) for name in variables) or (Path.home() / ".config" / "anthropic").is_dir()


def claude_parts(case: Case):
    from .claude_backend import ClaudeExtractor, ClaudeGateway, ClaudeReasoner

    gateway = ClaudeGateway()
    return ClaudeExtractor(case, gateway), ClaudeReasoner(gateway)


def groq_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def groq_parts(case: Case):
    from .groq_backend import GroqExtractor, GroqGateway, GroqReasoner

    gateway = GroqGateway()
    return GroqExtractor(case, gateway), GroqReasoner(gateway)


@dataclass
class Backends:
    """Separated so tests can supply scripted models."""

    claude_factory: Callable = claude_parts
    claude_available: Callable[[], bool] = claude_configured
    groq_factory: Callable = groq_parts
    groq_available: Callable[[], bool] = groq_configured

    def build(self, backend: str, case: Case):
        if backend == OFFLINE:
            return HeuristicExtractor(case), RuleReasoner()
        if backend == CLAUDE:
            if not self.claude_available():
                raise Conflict(NO_KEY_MESSAGE)
            return self.claude_factory(case)
        if backend == GROQ:
            if not self.groq_available():
                raise Conflict(NO_GROQ_KEY_MESSAGE)
            return self.groq_factory(case)
        raise NotFound(f"Unknown backend: {backend}")


class Workspace:
    def __init__(self, case_dir: Path, data_dir: Path, backends: Backends, builtin: bool):
        self.case_dir = case_dir
        self.builtin = builtin
        self.case = load_case(case_dir)
        self.backends = backends
        self.session_file = data_dir / "sessions" / f"{self.case.id}.json"
        self.status = "idle"  # idle, running, failed or done
        self.error = ""
        self.activity = ""
        self.backend = OFFLINE
        self.inv: Investigation | None = None
        self.questions: list[Answer] = []
        self._guard = threading.Lock()
        self._thread: threading.Thread | None = None
        self._browse_store = EvidenceStore(self.case)
        self._load()

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self.session_file.exists():
            return
        try:
            data = json.loads(self.session_file.read_text(encoding="utf-8"))
            self.inv = Investigation.restore(self.case, data["investigation"])
            self.backend = data["backend"]
            self.questions = [Answer.model_validate(q) for q in data["questions"]]
        except (ValueError, KeyError, TypeError) as error:
            logger.warning("Ignoring unreadable session for %s: %s", self.case.id, error)
            self.inv = None
            return
        self.status = "done" if self.inv.conclusion else "idle"

    def _save(self) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "investigation": self.inv.snapshot(),
            "questions": [q.model_dump() for q in self.questions],
        }
        temporary = self.session_file.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.session_file)

    def reset(self) -> None:
        self._require_idle()
        self.inv = None
        self.questions = []
        self.status, self.error, self.activity = "idle", "", ""
        self.session_file.unlink(missing_ok=True)

    # --- runs ---------------------------------------------------------------

    def start_run(self, backend: str, auto_leads: bool) -> None:
        extractor, reasoner = self.backends.build(backend, self.case)

        def work() -> None:
            self.inv = Investigation(self.case, extractor, ingest=False)
            self.questions = []
            self.activity = "Reading the evidence"
            self.inv.ingest_open_documents()
            self.activity = "Investigating"
            reasoner.investigate(self.inv, auto_leads=auto_leads)

        self.backend = backend
        self._launch(work)

    def uncover(self, lead_id: str) -> None:
        if self.inv is None or not self.inv.rounds:
            raise Conflict("Run an investigation before opening leads.")
        if lead_id not in {lead.id for lead in self.inv.pending_leads()}:
            raise NotFound(f"{lead_id} is not a sealed lead.")
        extractor, reasoner = self.backends.build(self.backend, self.case)

        def work() -> None:
            self.inv.extractor = extractor
            self.activity = f"Opening lead {lead_id}"
            self.inv.uncover(lead_id, "Requested by the investigator.")
            self.activity = "Reassessing the case"
            reasoner.investigate(self.inv, auto_leads=False)

        self._launch(work)

    def ask(self, question: str) -> Answer:
        self._require_idle()
        if self.inv is None or not self.inv.rounds:
            raise Conflict("Run an investigation before asking questions.")
        _, reasoner = self.backends.build(self.backend, self.case)
        try:
            answer = reasoner.answer(self.inv, question)
        except Exception as error:  # show the reason rather than a bare server error
            raise Conflict(describe_failure(error)) from error
        self.questions.append(answer)
        self._save()
        return answer

    def wait(self, timeout: float = 60) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _require_idle(self) -> None:
        if self.status == "running":
            raise Conflict("This case has a run in progress.")

    def _launch(self, work: Callable[[], None]) -> None:
        with self._guard:
            self._require_idle()
            self.status, self.error = "running", ""

        def runner() -> None:
            try:
                work()
                self._save()
                self.status = "done"
            except Exception as error:  # a run must always end in a visible state
                self.error = describe_failure(error)
                self.status = "failed"
            finally:
                self.activity = ""

        self._thread = threading.Thread(target=runner, daemon=True, name=f"run-{self.case.id}")
        self._thread.start()

    # --- reading ------------------------------------------------------------

    @property
    def store(self) -> EvidenceStore:
        return self.inv.store if self.inv else self._browse_store

    def summary(self) -> dict:
        conclusion = self.inv.conclusion if self.inv else None
        return {
            "id": self.case.id,
            "title": self.case.title,
            "difficulty": self.case.difficulty,
            "briefing": self.case.briefing,
            "documents": len(self.case.documents),
            "leads": len(self.case.leads),
            "suspects": len(self.case.suspects),
            "builtin": self.builtin,
            "status": self.status,
            "culprit": conclusion.culprit if conclusion else None,
            "confidence": conclusion.confidence.value if conclusion else None,
        }

    def overview(self) -> dict:
        case = self.case
        return self.summary() | {
            "crime": case.crime.model_dump(),
            "suspect_list": [p.model_dump() for p in case.suspects],
            "people": [p.model_dump() for p in case.people],
            "locations": [loc.model_dump() for loc in case.locations],
            "has_answer_key": has_solution(self.case_dir),
        }

    def documents(self) -> list[dict]:
        return [_document_row(doc) for doc in self.store.documents.values()]

    def document(self, doc_id: str) -> dict:
        doc = self.store.documents.get(doc_id)
        if doc is None:
            raise NotFound(f"{doc_id} is not an opened document.")
        return _document_row(doc) | {
            "speaker": doc.speaker,
            "device": doc.device,
            "passages": [p.model_dump() for p in split_passages(doc)],
        }

    def search(self, query: str) -> list[dict]:
        store = self.store
        return [
            {"passage_id": p.id, "doc_id": p.doc_id, "title": store.documents[p.doc_id].title, "text": p.text, "score": s}
            for p, s in store.search(query, k=12)
        ]

    def state(self) -> dict:
        inv = self.inv
        state = {
            "status": self.status,
            "activity": self.activity,
            "error": self.error,
            "backend": self.backend,
            "questions": [q.model_dump(mode="json") for q in self.questions],
            "pending_leads": [l.model_dump() for l in (inv.pending_leads() if inv else self.case.leads)],
            "opened_leads": list(inv.uncovered) if inv else [],
            "steps": [s.model_dump(mode="json") for s in list(inv.steps)] if inv else [],
            "rounds": [r.model_dump(mode="json") for r in list(inv.rounds)] if inv else [],
            "conclusion": None,
            "findings": [],
            "suspects": [],
            "timeline": [],
            "contradictions": [],
            "links": [],
            "defeaters": [],
            "compromised": {},
            "rejections": [],
            "finding_rejections": [],
        }
        if inv is None:
            return state
        state["rejections"] = [r.model_dump() for r in list(inv.rejections)]
        state["finding_rejections"] = list(inv.finding_rejections)
        analysis = inv.analysis
        if analysis is not None and self.status != "running":
            state |= {
                "conclusion": inv.conclusion.model_dump(mode="json") if inv.conclusion else None,
                "findings": [f.model_dump(mode="json") for f in inv.findings],
                "suspects": [s.model_dump(mode="json") for s in inv.suspects],
                "timeline": [
                    e.model_dump(mode="json") | {"refuted": index in analysis.refuted, "tier_name": e.tier.name.lower()}
                    for index, e in enumerate(analysis.timeline)
                ],
                "contradictions": [c.model_dump(mode="json") for c in analysis.contradictions],
                "links": [link.model_dump(mode="json") for link in analysis.links],
                "defeaters": [d.model_dump(mode="json") for d in analysis.defeaters],
                "compromised": analysis.compromised,
            }
        return state

    def report(self) -> str:
        if self.inv is None or self.inv.conclusion is None:
            raise Conflict("There is no finished investigation to report.")
        return render_report(self.inv)

    def evaluate(self) -> dict:
        if not has_solution(self.case_dir):
            raise NotFound("This case has no answer key.")
        if self.inv is None or self.inv.conclusion is None:
            raise Conflict("Finish an investigation before checking it against the answer key.")
        return score(self.inv, load_solution(self.case_dir)).as_dict()


class WorkspaceManager:
    def __init__(self, data_dir: Path, cases_dir: Path = BUILTIN_CASES, backends: Backends | None = None):
        self.data_dir = data_dir
        self.cases_dir = cases_dir
        self.custom_dir = data_dir / "cases"
        self.backends = backends or Backends()
        self.workspaces: dict[str, Workspace] = {}
        self._discover()

    def _discover(self) -> None:
        for root, builtin in ((self.cases_dir, True), (self.custom_dir, False)):
            if not root.exists():
                continue
            for folder in sorted(root.iterdir()):
                if (folder / "case.json").exists():
                    try:
                        self._add(folder, builtin)
                    except (ValueError, KeyError, OSError) as error:
                        logger.warning("Skipping unreadable case %s: %s", folder.name, error)

    def _add(self, folder: Path, builtin: bool) -> Workspace:
        workspace = Workspace(folder, self.data_dir, self.backends, builtin)
        self.workspaces[workspace.case.id] = workspace
        return workspace

    def get(self, case_id: str) -> Workspace:
        if case_id not in self.workspaces:
            raise NotFound(f"No case called {case_id}.")
        return self.workspaces[case_id]

    def create(self, draft: CaseDraft) -> Workspace:
        case_id = create_from_draft(draft, self.custom_dir, set(self.workspaces))
        return self._add(self.custom_dir / case_id, False)

    def import_zip(self, data: bytes) -> Workspace:
        case_id = import_archive(data, self.custom_dir, set(self.workspaces))
        return self._add(self.custom_dir / case_id, False)

    def delete(self, case_id: str) -> None:
        workspace = self.get(case_id)
        if workspace.builtin:
            raise Conflict("Built-in cases cannot be deleted.")
        workspace.reset()
        shutil.rmtree(workspace.case_dir)
        del self.workspaces[case_id]


def describe_failure(error: Exception) -> str:
    """Turn an exception into a message an investigator can act on."""
    import anthropic
    import groq

    from .claude_backend import RefusedError

    if isinstance(error, RefusedError):
        return f"The model declined this request. Details: {error}"
    if isinstance(error, groq.AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY and restart the server."
    if isinstance(error, groq.RateLimitError):
        return "Groq's rate limit was reached. Wait a moment and try again."
    if isinstance(error, groq.APIConnectionError):
        return "Could not reach the Groq API. Check your network connection and try again."
    if isinstance(error, groq.APIStatusError):
        return f"The Groq API returned an error ({error.status_code}): {error.message}"
    if isinstance(error, anthropic.AuthenticationError):
        return "Claude rejected the API key. Check ANTHROPIC_API_KEY and restart the server."
    if isinstance(error, anthropic.RateLimitError):
        return "Claude's rate limit was reached. Wait a moment and try again."
    if isinstance(error, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check your network connection and try again."
    if isinstance(error, anthropic.APIStatusError):
        return f"The Claude API returned an error ({error.status_code}): {error.message}"
    return str(error) or type(error).__name__


def _document_row(doc: Document) -> dict:
    return {"id": doc.id, "type": doc.type, "title": doc.title, "date": doc.date, "tier": int(doc.tier), "tier_name": doc.tier.name.lower()}
