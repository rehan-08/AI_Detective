"""The anti-hallucination gate.

Nothing an extractor or the investigator says enters the case until it is checked
against the source text. The rules are mechanical on purpose:

* a citation must point at a real passage and quote it verbatim;
* a person, location or time asserted by an item must appear in the cited passage
  (or, for first-person statements, be the document's speaker);
* a "fact" must rest on a record, a document, or two independent sources.
"""
from __future__ import annotations

import re

from .models import (
    Case,
    Citation,
    DocExtraction,
    Document,
    ExtractionRejection,
    Finding,
    Status,
    Tier,
)
from .store import EvidenceStore


def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().casefold()


class Grounding:
    def __init__(self, case: Case, store: EvidenceStore):
        self.case = case
        self.store = store
        self._people = {p.name: [p.name, *p.aliases] for p in case.everyone()}
        self._places = {loc.name: [loc.name, *loc.aliases] for loc in case.locations}

    def check_citation(self, citation: Citation) -> tuple[Citation | None, str]:
        """Return a verified citation (repairing a wrong passage ID) or a reason."""
        quote = normalize(citation.quote)
        if len(quote) < 4:
            return None, "quote is empty or too short to verify"
        passage = self.store.passages.get(citation.passage_id)
        if passage and quote in normalize(passage.text):
            return citation, ""
        doc_id = citation.passage_id.split(":")[0]
        for candidate in self.store.doc_passages(doc_id):
            if quote in normalize(candidate.text):
                return Citation(passage_id=candidate.id, quote=citation.quote), ""
        if passage is None and doc_id not in self.store.documents:
            return None, f"no such source: {citation.passage_id}"
        return None, f"quote not found in {citation.passage_id}: {citation.quote!r}"

    def filter_extraction(self, extraction: DocExtraction) -> tuple[DocExtraction, list[ExtractionRejection]]:
        """Admit only the items that survive verification."""
        rejected: list[ExtractionRejection] = []
        kept = extraction.model_copy(deep=True)
        kept.entities = self._keep(extraction, "entity", extraction.entities, rejected)
        kept.events = self._keep(extraction, "event", extraction.events, rejected, self._event_problem)
        kept.signals = self._keep(extraction, "signal", extraction.signals, rejected, self._signal_problem)
        kept.defeaters = self._keep(extraction, "defeater", extraction.defeaters, rejected, self._defeater_problem)
        kept.credentials = self._keep(extraction, "credential", extraction.credentials, rejected)
        kept.identifiers = self._keep(extraction, "identifier", extraction.identifiers, rejected, self._identifier_problem)
        return kept, rejected

    def check_finding(self, finding: Finding) -> tuple[Finding | None, str]:
        """Verify citations and enforce what the status label is allowed to claim."""
        verified, problems = [], []
        for citation in finding.citations:
            good, reason = self.check_citation(citation)
            if good:
                verified.append(good)
            else:
                problems.append(reason)
        if problems:
            return None, "; ".join(problems)
        if finding.status != Status.SPECULATION and not verified:
            return None, f"a {finding.status.value} must cite evidence"
        result = finding.model_copy(update={"citations": verified})
        if result.status == Status.FACT and not self._can_be_fact(verified):
            note = "Downgraded from fact: rests on a single low-tier or interested source."
            result = result.model_copy(
                update={"status": Status.ASSUMPTION, "rationale": f"{finding.rationale} {note}".strip()}
            )
        return result, ""

    def _can_be_fact(self, citations: list[Citation]) -> bool:
        docs = {c.passage_id.split(":")[0] for c in citations}
        tiers = [self.store.documents[d].tier for d in docs]
        return any(t >= Tier.DOCUMENT for t in tiers) or len([t for t in tiers if t >= Tier.TESTIMONY]) >= 2

    def _keep(self, extraction, item_type, items, rejected, problem=None) -> list:
        kept = []
        for item in items:
            citation, reason = self.check_citation(item.citation)
            if citation and problem:
                reason = problem(item.model_copy(update={"citation": citation}), extraction.doc_id)
            if citation and not reason:
                kept.append(item.model_copy(update={"citation": citation}))
            else:
                rejected.append(
                    ExtractionRejection(
                        doc_id=extraction.doc_id, item_type=item_type, reason=reason, content=item.model_dump_json()
                    )
                )
        return kept

    def _source(self, doc_id: str) -> Document:
        return self.store.documents[doc_id]

    def _passage_text(self, item) -> str:
        return normalize(self.store.passages[item.citation.passage_id].text)

    def _mentions(self, text: str, name: str, table: dict[str, list[str]]) -> bool:
        return any(normalize(alias) in text for alias in table.get(name, [name]))

    def _person_problem(self, item, person: str, doc_id: str) -> str:
        if person not in self._people:
            return f"unknown person: {person}"
        speaker = self._source(doc_id).speaker
        if person == speaker or self._mentions(self._passage_text(item), person, self._people):
            return ""
        return f"{person} is not mentioned in the cited passage"

    def _event_problem(self, event, doc_id: str) -> str:
        if event.person:
            reason = self._person_problem(event, event.person, doc_id)
            if reason:
                return reason
        if event.credential and normalize(event.credential) not in self._passage_text(event):
            return f"credential {event.credential} is not in the cited passage"
        if event.location and event.location not in self._places and not event.location.startswith("Home of"):
            return f"unknown location: {event.location}"
        if event.start and not time_in_text(event.start, self._passage_text(event)):
            return f"time {event.start} does not appear in the cited passage"
        return ""

    def _signal_problem(self, signal, doc_id: str) -> str:
        return self._person_problem(signal, signal.person, doc_id)

    def _defeater_problem(self, defeater, doc_id: str) -> str:
        text = self._passage_text(defeater)
        if defeater.minutes is not None and str(abs(defeater.minutes)) not in text:
            return f"offset of {defeater.minutes} minutes is not in the cited passage"
        return ""

    def _identifier_problem(self, identifier, doc_id: str) -> str:
        if normalize(identifier.value) not in self._passage_text(identifier):
            return f"identifier {identifier.value} is not in the cited passage"
        return ""


def time_in_text(iso: str, text: str) -> bool:
    """True when a time written as HH:MM, H:MM or 'H pm' occurs in the text."""
    clock = iso.split("T")[-1]
    hour, minute = int(clock[:2]), clock[3:5]
    twelve = hour % 12 or 12
    meridian = "pm" if hour >= 12 else "am"
    forms = [clock[:5], f"{hour}:{minute}", f"{twelve}:{minute}", f"{twelve}.{minute}"]
    if minute == "00":
        forms += [f"{twelve} {meridian}", f"{twelve}{meridian}", f"{twelve} o'clock"]
    if minute == "30":
        forms += ["half past"]
    return any(form in text for form in forms)
