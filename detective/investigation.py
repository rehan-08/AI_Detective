"""The investigation: evidence state, rounds of reassessment and the reasoning log."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .analysis import Analysis, analyse
from .engine import Engine
from .grounding import Grounding
from .models import (
    Answer,
    Assessment,
    Case,
    Citation,
    Conclusion,
    Contradiction,
    DocExtraction,
    Document,
    ExtractionRejection,
    Finding,
    Lead,
    ReasoningStep,
    Status,
    SuspectAssessment,
)
from .store import EvidenceStore


class Extractor(Protocol):
    name: str

    def extract(self, doc: Document) -> DocExtraction: ...


class Reasoner(Protocol):
    name: str

    def investigate(self, investigation: "Investigation", auto_leads: bool = True) -> Conclusion: ...

    def answer(self, investigation: "Investigation", question: str) -> Answer: ...


class Investigation:
    def __init__(self, case: Case, extractor: Extractor | None, ingest: bool = True):
        self.case = case
        self.extractor = extractor
        self.store = EvidenceStore(case)
        self.grounding = Grounding(case, self.store)
        self.extractions: dict[str, DocExtraction] = {}
        self.rejections: list[ExtractionRejection] = []
        self.steps: list[ReasoningStep] = []
        self.findings: list[Finding] = []
        self.finding_rejections: list[str] = []
        self.semantic_contradictions: list[Contradiction] = []
        self.rounds: list[Assessment] = []
        self.uncovered: list[str] = []
        self.conclusion: Conclusion | None = None
        self.analysis: Analysis | None = None
        self.suspects: list[SuspectAssessment] = []
        if ingest:
            self.ingest_open_documents()

    # --- persistence --------------------------------------------------------

    def snapshot(self) -> dict:
        """Everything needed to resume without repeating any model calls."""
        return {
            "uncovered": self.uncovered,
            "extractions": [e.model_dump() for e in self.extractions.values()],
            "rejections": [r.model_dump() for r in self.rejections],
            "steps": [s.model_dump() for s in self.steps],
            "findings": [f.model_dump() for f in self.findings],
            "finding_rejections": self.finding_rejections,
            "semantic_contradictions": [c.model_dump() for c in self.semantic_contradictions],
            "rounds": [r.model_dump() for r in self.rounds],
            "conclusion": self.conclusion.model_dump() if self.conclusion else None,
        }

    @classmethod
    def restore(cls, case: Case, data: dict, extractor: Extractor | None = None) -> "Investigation":
        inv = cls(case, extractor, ingest=False)
        inv.uncovered = list(data["uncovered"])
        for lead_id in inv.uncovered:
            inv.store.add_document(case.sealed[lead_id])
        inv.extractions = {e["doc_id"]: DocExtraction.model_validate(e) for e in data["extractions"]}
        inv.rejections = [ExtractionRejection.model_validate(r) for r in data["rejections"]]
        inv.steps = [ReasoningStep.model_validate(s) for s in data["steps"]]
        inv.findings = [Finding.model_validate(f) for f in data["findings"]]
        inv.finding_rejections = list(data["finding_rejections"])
        inv.semantic_contradictions = [Contradiction.model_validate(c) for c in data["semantic_contradictions"]]
        inv.rounds = [Assessment.model_validate(r) for r in data["rounds"]]
        inv.conclusion = Conclusion.model_validate(data["conclusion"]) if data["conclusion"] else None
        if inv.rounds:
            inv._recompute()
        return inv

    # --- evidence -----------------------------------------------------------

    def ingest_open_documents(self) -> None:
        self._ingest(self.case.documents)

    def _ingest(self, docs: list[Document]) -> None:
        workers = getattr(self.extractor, "workers", 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            raw = list(pool.map(self.extractor.extract, docs))
        for doc, extraction in zip(docs, raw):
            kept, rejected = self.grounding.filter_extraction(extraction.model_copy(update={"doc_id": doc.id}))
            self.extractions[doc.id] = kept
            self.rejections.extend(rejected)
            self.log(
                "extract",
                f"{doc.id} ({doc.type}): {len(kept.events)} events, {len(kept.signals)} signals, "
                f"{len(kept.defeaters)} defeaters, {len(kept.identifiers)} identifiers admitted; {len(rejected)} rejected.",
            )

    def pending_leads(self) -> list[Lead]:
        return [lead for lead in self.case.leads if lead.id not in self.uncovered]

    def uncover(self, lead_id: str, reason: str = "") -> Assessment:
        """Open a sealed lead, then re-run the analysis from scratch."""
        lead = next((l for l in self.pending_leads() if l.id == lead_id), None)
        if lead is None:
            raise KeyError(f"no pending lead {lead_id}")
        doc = self.case.sealed[lead_id]
        self.store.add_document(doc)
        self.uncovered.append(lead_id)
        self.log("uncover", f"Opened lead {lead_id}: {lead.title} -> {doc.id}. {reason}".strip())
        self._ingest([doc])
        return self.reassess(f"Uncovered {doc.id}: {doc.title}")

    # --- reasoning ----------------------------------------------------------

    def begin(self) -> Assessment:
        """Run the first review, unless a restored session already has one."""
        return self.rounds[-1] if self.rounds else self.reassess("Initial review of the open case file")

    def _recompute(self) -> Engine:
        extractions = list(self.extractions.values())
        self.analysis = analyse(self.case, self.store, extractions, self.semantic_contradictions)
        engine = Engine(self.case, self.store, self.analysis)
        self.suspects = engine.assess_all()
        return engine

    def reassess(self, trigger: str) -> Assessment:
        engine = self._recompute()
        assessment = engine.judge(self.suspects, len(self.rounds), trigger)
        previous = self.rounds[-1] if self.rounds else None
        self.rounds.append(assessment)
        self.log("assess", _describe(assessment, previous))
        return assessment

    def add_finding(self, finding: Finding) -> tuple[Finding | None, str]:
        checked, reason = self.grounding.check_finding(finding)
        if checked is None:
            self.finding_rejections.append(f"{finding.statement} -> {reason}")
            self.log("reject_finding", f"{finding.statement} -> {reason}")
            return None, reason
        checked.id = f"F{len(self.findings) + 1}"
        self.findings.append(checked)
        self.log("finding", f"[{checked.status.value}] {checked.statement}", checked.citations)
        return checked, ""

    def log(self, action: str, observation: str, citations: list[Citation] | None = None) -> None:
        self.steps.append(
            ReasoningStep(round=len(self.rounds), action=action, observation=observation, citations=citations or [])
        )

    def cited_documents(self, status: Status | None = None) -> set[str]:
        found = self.findings if status is None else [f for f in self.findings if f.status == status]
        return {c.passage_id.split(":")[0] for f in found for c in f.citations}

    def all_citations(self) -> list[Citation]:
        cited = [c for f in self.findings for c in f.citations]
        if self.analysis:
            cited += [c for x in self.analysis.contradictions for c in x.citations]
        return cited


def _describe(assessment: Assessment, previous: Assessment | None) -> str:
    text = (
        f"Round {assessment.round} ({assessment.trigger}): leader {assessment.leader}, "
        f"confidence {assessment.confidence.value}."
    )
    if previous and previous.leader != assessment.leader:
        text += f" The leader changed from {previous.leader}."
    return text
