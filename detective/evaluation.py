"""Score a finished investigation against a case's predefined solution.

This is the only module that reads solution.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .investigation import Investigation
from .models import Confidence, Status

CONFIDENCE_ORDER = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]


@dataclass
class CaseScore:
    case: str
    named: str | None
    expected: str
    culprit_correct: bool
    confidence: str
    confidence_ok: bool
    accomplices_correct: bool
    cleared_correct: bool
    contradiction_recall: float
    timeline_recall: float
    key_evidence_recall: float
    red_herrings_clean: bool
    citation_validity: float
    retrieval_recall: float
    first_round_ok: bool
    face_value_ok: bool
    misses: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.culprit_correct and self.confidence_ok and self.accomplices_correct and self.cleared_correct
            and self.red_herrings_clean and self.citation_validity == 1.0 and self.first_round_ok and self.face_value_ok
        )

    def as_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}


def has_solution(case_dir: str | Path) -> bool:
    return (Path(case_dir) / "solution.json").exists()


def load_solution(case_dir: str | Path) -> dict:
    return json.loads((Path(case_dir) / "solution.json").read_text(encoding="utf-8"))


def score(inv: Investigation, solution: dict) -> CaseScore:
    conclusion = inv.conclusion
    misses: list[str] = []
    contradiction = _recall(
        solution["expected_contradictions"],
        lambda want: any(
            c.person == want["person"] and set(want["documents"]) <= set(c.documents)
            for c in inv.analysis.contradictions
        ),
        misses,
        "contradiction",
    )
    timeline = _recall(
        solution["expected_timeline"],
        lambda want: any(
            e.person == want["person"] and e.doc_id == want["document"] and want["time"] in (e.time, e.end)
            for e in inv.analysis.timeline
        ),
        misses,
        "timeline",
    )
    wanted = solution["key_evidence"] + [
        doc for doc in solution["key_evidence_after_leads"] if inv.uncovered or doc in inv.store.documents
    ]
    cited = inv.cited_documents() | {doc for c in inv.analysis.contradictions for doc in c.documents}
    key = _recall(wanted, lambda doc: doc in cited, misses, "key evidence")
    support = inv.cited_documents(Status.FACT) | inv.cited_documents(Status.ASSUMPTION)
    herring_hits = support & set(solution["red_herring_documents"])
    if herring_hits:
        misses.append(f"red herring used as support: {sorted(herring_hits)}")
    citations = inv.all_citations()
    valid = sum(1 for c in citations if inv.grounding.check_citation(c)[0]) / len(citations) if citations else 0.0
    retrieval = _recall(
        [p for p in solution["retrieval_probes"] if all(d in inv.store.documents for d in p["expected"])],
        lambda probe: {p.doc_id for p, _ in inv.store.search(probe["query"], k=5)} & set(probe["expected"]),
        misses,
        "retrieval",
    )
    final_names = set(conclusion.accomplices)
    expected_first = solution.get("first_round_leader")
    expected_face = solution.get("face_value_leader")
    required = CONFIDENCE_ORDER.index(Confidence(solution["minimum_confidence"]))
    excluded = {s.name for s in inv.suspects if s.opportunity == "excluded"}
    culprit_correct = conclusion.culprit == solution["culprit"]
    return CaseScore(
        case=inv.case.id,
        named=conclusion.culprit,
        expected=solution["culprit"],
        culprit_correct=culprit_correct,
        confidence=conclusion.confidence.value,
        confidence_ok=CONFIDENCE_ORDER.index(conclusion.confidence) >= required,
        accomplices_correct=final_names == set(solution["accomplices"]),
        cleared_correct=conclusion.culprit not in solution["cleared"] and solution["culprit"] not in excluded,
        contradiction_recall=contradiction,
        timeline_recall=timeline,
        key_evidence_recall=key,
        red_herrings_clean=not herring_hits,
        citation_validity=valid,
        retrieval_recall=retrieval,
        first_round_ok=expected_first is None or inv.rounds[0].leader == expected_first,
        face_value_ok=expected_face is None or inv.rounds[0].face_value_leader == expected_face,
        misses=misses,
    )


def _recall(wanted: list, found, misses: list[str], label: str) -> float:
    if not wanted:
        return 1.0
    hits = 0
    for item in wanted:
        if found(item):
            hits += 1
        else:
            misses.append(f"{label}: {item}")
    return hits / len(wanted)


def format_table(scores: list[CaseScore]) -> str:
    rows = [
        ("Culprit", lambda s: f"{s.named} ({'correct' if s.culprit_correct else 'WRONG'})"),
        ("Confidence", lambda s: f"{s.confidence} ({'ok' if s.confidence_ok else 'too low'})"),
        ("Accomplices", lambda s: "ok" if s.accomplices_correct else "WRONG"),
        ("Contradiction recall", lambda s: f"{s.contradiction_recall:.0%}"),
        ("Timeline recall", lambda s: f"{s.timeline_recall:.0%}"),
        ("Key evidence cited", lambda s: f"{s.key_evidence_recall:.0%}"),
        ("Retrieval recall@5", lambda s: f"{s.retrieval_recall:.0%}"),
        ("Citation validity", lambda s: f"{s.citation_validity:.0%}"),
        ("Red herrings avoided", lambda s: "yes" if s.red_herrings_clean else "NO"),
        ("Overall", lambda s: "PASS" if s.passed else "FAIL"),
    ]
    header = "| Metric | " + " | ".join(s.case for s in scores) + " |"
    divider = "|---|" + "---|" * len(scores)
    body = ["| " + name + " | " + " | ".join(render(s) for s in scores) + " |" for name, render in rows]
    return "\n".join([header, divider, *body])
