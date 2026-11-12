"""Tools the Claude investigator uses. Each one reads or writes guarded case state."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .investigation import Investigation
from .models import Citation, Conclusion, Confidence, Contradiction, Finding, Status


class ToolError(Exception):
    """Returned to the model as an is_error tool result so it can correct itself."""


_CITATIONS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "passage_id": {"type": "string", "description": "Passage ID such as D2:p4"},
            "quote": {"type": "string", "description": "Text copied verbatim from that passage"},
        },
        "required": ["passage_id", "quote"],
        "additionalProperties": False,
    },
}


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_evidence",
        "description": "BM25 search over every opened document. Returns passages with IDs for citation. Search for specific names, identifiers, places and times; run several searches rather than one broad one.",
        "input_schema": _schema({"query": {"type": "string"}}),
    },
    {
        "name": "read_document",
        "description": "Read one opened document in full, with passage IDs, its type and its source tier (0 anonymous, 1 testimony, 2 document, 3 record).",
        "input_schema": _schema({"doc_id": {"type": "string"}}),
    },
    {
        "name": "get_timeline",
        "description": "The verified timeline, with clock corrections applied and credential use attributed. Pass an empty string for everyone or a canonical name for one person.",
        "input_schema": _schema({"person": {"type": "string"}}),
    },
    {
        "name": "get_contradictions",
        "description": "Contradictions found by the deterministic detector, with how each was resolved.",
        "input_schema": _schema({}),
    },
    {
        "name": "get_suspect_matrix",
        "description": "The rule engine's motive, means, opportunity and alibi matrix with score breakdowns. Use it as a cross-check: it only sees what extraction captured.",
        "input_schema": _schema({}),
    },
    {
        "name": "uncover_lead",
        "description": "Open a sealed lead listed in the brief. Its document joins the evidence and the analysis is re-run. Give the reason this lead is worth following.",
        "input_schema": _schema({"lead_id": {"type": "string"}, "reason": {"type": "string"}}),
    },
    {
        "name": "record_finding",
        "description": "Record one conclusion in the case notes. status: fact (directly supported by a cited source), assumption (a reasonable inference that goes beyond the source), speculation (a possibility with thin support). A citation that cannot be verified verbatim is rejected. A fact resting on one uncorroborated statement is downgraded.",
        "input_schema": _schema(
            {
                "statement": {"type": "string"},
                "status": {"type": "string", "enum": ["fact", "assumption", "speculation"]},
                "citations": _CITATIONS,
                "rationale": {"type": "string", "description": "How the cited evidence supports the statement"},
            }
        ),
    },
    {
        "name": "record_contradiction",
        "description": "Record a contradiction the detector missed, such as a clash in meaning rather than in time and place. Needs at least two verified citations from different documents.",
        "input_schema": _schema(
            {
                "person": {"type": "string"},
                "summary": {"type": "string"},
                "resolution": {"type": "string", "description": "Which source prevails and why, or that it is unresolved"},
                "discredited_speaker": {"type": "string", "description": "Who gave the false account, or an empty string"},
                "citations": _CITATIONS,
            }
        ),
    },
    {
        "name": "submit_conclusion",
        "description": "File the final conclusion. Call this once, last. supporting_finding_ids must name recorded findings of status fact or assumption. Every other suspect needs an entry in eliminated.",
        "input_schema": _schema(
            {
                "culprit": {"type": "string"},
                "accomplices": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                "confidence_explanation": {"type": "string"},
                "narrative": {"type": "string", "description": "Short reconstruction of the crime"},
                "supporting_finding_ids": {"type": "array", "items": {"type": "string"}},
                "eliminated": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "reason": {"type": "string"}},
                        "required": ["name", "reason"],
                        "additionalProperties": False,
                    },
                },
                "disagreement_with_engine": {
                    "type": "string",
                    "description": "If your culprit differs from the rule engine's leader, explain why. Otherwise an empty string.",
                },
            }
        ),
    },
]


class InvestigatorTools:
    def __init__(self, investigation: Investigation, allow_leads: bool = True):
        self.inv = investigation
        self.allow_leads = allow_leads
        self.engine_challenged = False

    def definitions(self) -> list[dict[str, Any]]:
        chosen = [t for t in TOOLS if self.allow_leads or t["name"] != "uncover_lead"]
        return [{**tool, "strict": True} for tool in chosen]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        handler = getattr(self, f"_{name}", None)
        if handler is None or name.startswith("_"):
            raise ToolError(f"unknown tool {name}")
        try:
            return json.dumps(handler(**arguments), ensure_ascii=False)
        except (TypeError, ValidationError) as error:
            raise ToolError(f"invalid arguments: {error}") from error

    def _search_evidence(self, query: str) -> dict:
        hits = self.inv.store.search(query, k=6)
        self.inv.log("search", f"Query {query!r} returned {[p.id for p, _ in hits]}")
        return {
            "results": [
                {"passage_id": p.id, "document": self.inv.store.documents[p.doc_id].title, "text": p.text, "score": s}
                for p, s in hits
            ]
        }

    def _read_document(self, doc_id: str) -> dict:
        doc = self.inv.store.documents.get(doc_id)
        if doc is None:
            raise ToolError(f"{doc_id} is not an opened document")
        self.inv.log("read", f"Read {doc_id}: {doc.title}")
        return {
            "id": doc.id,
            "type": doc.type,
            "tier": int(doc.tier),
            "title": doc.title,
            "speaker": doc.speaker,
            "device": doc.device,
            "passages": [{"passage_id": p.id, "text": p.text} for p in self.inv.store.doc_passages(doc_id)],
        }

    def _get_timeline(self, person: str) -> dict:
        entries = [e for e in self.inv.analysis.timeline if not person or e.person == person]
        return {"entries": [e.model_dump(exclude={"citation"}) | {"passage_id": e.citation.passage_id} for e in entries]}

    def _get_contradictions(self) -> dict:
        return {"contradictions": [c.model_dump() for c in self.inv.analysis.contradictions]}

    def _get_suspect_matrix(self) -> dict:
        return {
            "suspects": [
                s.model_dump(include={"name", "score", "breakdown", "opportunity", "opportunity_note", "alibi", "alibi_note", "lies"})
                | {
                    "motives": [m.detail for m in s.motives],
                    "means": [m.detail for m in s.means],
                    "suspicious": [m.detail for m in s.suspicious],
                    "links": [link.path for link in s.links],
                }
                for s in self.inv.suspects
            ],
            "assessment": self.inv.rounds[-1].model_dump(),
        }

    def _uncover_lead(self, lead_id: str, reason: str) -> dict:
        if not self.allow_leads:
            raise ToolError("leads are disabled for this run")
        try:
            assessment = self.inv.uncover(lead_id, reason)
        except KeyError as error:
            raise ToolError(str(error)) from error
        doc = self.inv.case.sealed[lead_id]
        return {"document": self._read_document(doc.id), "assessment": assessment.model_dump()}

    def _record_finding(self, statement: str, status: str, citations: list[dict], rationale: str) -> dict:
        finding = Finding(statement=statement, status=Status(status), citations=[Citation(**c) for c in citations], rationale=rationale)
        accepted, reason = self.inv.add_finding(finding)
        if accepted is None:
            raise ToolError(f"Finding rejected: {reason}")
        return {"id": accepted.id, "status": accepted.status.value, "note": accepted.rationale}

    def _record_contradiction(self, person: str, summary: str, resolution: str, discredited_speaker: str, citations: list[dict]) -> dict:
        verified = []
        for raw in citations:
            good, reason = self.inv.grounding.check_citation(Citation(**raw))
            if good is None:
                raise ToolError(f"Contradiction rejected: {reason}")
            verified.append(good)
        documents = sorted({c.passage_id.split(":")[0] for c in verified})
        if len(documents) < 2:
            raise ToolError("Contradiction rejected: it needs citations from two different documents")
        item = Contradiction(
            id=f"S{len(self.inv.semantic_contradictions) + 1}",
            person=person,
            summary=summary,
            documents=documents,
            citations=verified,
            kind="semantic",
            resolution=resolution,
            discredited_speaker=discredited_speaker or None,
        )
        self.inv.semantic_contradictions.append(item)
        self.inv.analysis.contradictions.append(item)
        self.inv.log("contradiction", summary, verified)
        return {"id": item.id}

    def _submit_conclusion(
        self, culprit, accomplices, confidence, confidence_explanation, narrative, supporting_finding_ids, eliminated, disagreement_with_engine
    ) -> dict:
        names = {p.name for p in self.inv.case.suspects}
        if culprit not in names:
            raise ToolError(f"{culprit} is not a suspect in this case")
        unknown = [a for a in accomplices if a not in names]
        if unknown:
            raise ToolError(f"unknown accomplices: {unknown}")
        by_id = {f.id: f for f in self.inv.findings}
        missing = [i for i in supporting_finding_ids if i not in by_id]
        if missing:
            raise ToolError(f"unknown findings: {missing}")
        chosen = [by_id[i] for i in supporting_finding_ids]
        if any(f.status == Status.SPECULATION for f in chosen):
            raise ToolError("speculation cannot support a conclusion; remove it or find evidence for it")
        if not any(f.status == Status.FACT for f in chosen):
            raise ToolError("a conclusion needs at least one supporting fact")
        accounted = {e["name"] for e in eliminated}
        if names - accounted - {culprit}:
            raise ToolError(f"explain why these suspects were ruled out or ranked lower: {sorted(names - accounted - {culprit})}")

        leader = self.inv.rounds[-1].leader
        agrees = culprit == leader
        if not agrees and not disagreement_with_engine.strip():
            raise ToolError(
                f"The rule engine ranks {leader} first. Fill disagreement_with_engine explaining, with evidence, "
                "why it is wrong, or revise your conclusion."
            )
        self.inv.conclusion = Conclusion(
            culprit=culprit,
            confidence=Confidence(confidence),
            confidence_explanation=confidence_explanation,
            narrative=narrative,
            accomplices=accomplices,
            supporting_findings=supporting_finding_ids,
            eliminated={e["name"]: e["reason"] for e in eliminated},
            engine_agrees=agrees,
            engine_note="" if agrees else f"Rule engine leader: {leader}. Investigator's reason: {disagreement_with_engine}",
        )
        self.inv.log("conclude", f"Named {culprit} at {confidence} confidence.")
        return {"status": "filed"}
