"""Offline reasoner: a fixed investigation policy with templated findings.

It drives the same evidence state and guards as the Claude reasoner, without a
model. Its value is reproducibility: every test can run with no credentials.
"""
from __future__ import annotations

import re

from .analysis import overlaps
from .engine import dedupe_citations
from .investigation import Investigation
from .models import Answer, Citation, Conclusion, Confidence, Finding, Lead, Status, SuspectAssessment, Tier
from .store import tokenize

MIN_ANSWER_SCORE = 2.0


class RuleReasoner:
    name = "rules"

    def investigate(self, inv: Investigation, auto_leads: bool = True) -> Conclusion:
        inv.begin()
        while auto_leads and inv.rounds[-1].confidence != Confidence.HIGH and inv.pending_leads():
            lead = pick_lead(inv)
            inv.uncover(lead.id, "Confidence is below HIGH, and this lead bears on an open question.")
        return self._conclude(inv)

    def _conclude(self, inv: Investigation) -> Conclusion:
        final = inv.rounds[-1]
        leader = next((s for s in inv.suspects if s.name == final.leader), None)
        inv.findings.clear()
        support = _write_findings(inv, leader)
        eliminated = {s.name: _elimination(s) for s in inv.suspects if s.name != final.leader}
        conclusion = Conclusion(
            culprit=final.leader,
            confidence=final.confidence,
            confidence_explanation=final.confidence_explanation,
            narrative=_narrative(inv, leader),
            accomplices=final.accomplices,
            supporting_findings=support,
            eliminated=eliminated,
        )
        inv.conclusion = conclusion
        return conclusion

    def answer(self, inv: Investigation, question: str) -> Answer:
        if inv.analysis is None:
            inv.reassess("Review before answering a question")
        hits = [(p, s) for p, s in inv.store.search(question, k=4) if s >= MIN_ANSWER_SCORE and _on_topic(question, p.text)]
        people = [s for s in inv.suspects if _mentions(question, inv, s.name)]
        if not hits and not people:
            return Answer(
                question=question,
                answer="The case file holds no evidence on this question.",
                status=Status.SPECULATION,
                grounded=False,
            )
        lines = [_profile(s) for s in people]
        lines += [f"{p.id}: {p.text}" for p, _ in hits]
        citations = [Citation(passage_id=p.id, quote=p.text) for p, _ in hits]
        for suspect in people:
            citations.extend(suspect.support[:3])
        return Answer(question=question, answer="\n".join(lines), status=Status.FACT, citations=dedupe_citations(citations))


def pick_lead(inv: Investigation) -> Lead:
    """Choose the sealed lead that best matches the open questions."""
    wanted = set(tokenize(" ".join(inv.rounds[-1].open_questions)))
    return max(inv.pending_leads(), key=lambda lead: len(wanted & set(tokenize(lead.title))))


def _write_findings(inv: Investigation, leader: SuspectAssessment | None) -> list[str]:
    if leader is None:
        return []
    support: list[str] = []

    def add(statement: str, status: Status, citations: list[Citation], rationale: str = "", key: bool = True) -> None:
        finding, _ = inv.add_finding(
            Finding(statement=statement, status=status, citations=dedupe_citations(citations), rationale=rationale)
        )
        if finding and key and finding.status != Status.SPECULATION:
            support.append(finding.id)

    name = leader.name
    window = inv.case.crime.window_start, inv.case.crime.window_end
    placed = [
        e for e in inv.analysis.timeline
        if e.person == name and e.present and e.tier >= Tier.RECORD and e.attribution != "disputed"
        and (overlaps(e.time, e.end, *window) or window[0] <= e.time <= window[1])
    ]
    for entry in placed:
        add(f"Record {entry.doc_id} places {name} at {entry.location} at {entry.time.split('T')[-1]}.", Status.FACT, [entry.citation])
        if entry.attribution == "credential":
            add(
                f"{entry.credential} was used by {name} personally.",
                Status.ASSUMPTION,
                [entry.citation],
                "A credential log shows the credential, not the hand that held it. Nothing in the file disputes this use.",
                key=False,
            )
    for defeater in inv.analysis.defeaters:
        add(f"Evidence was undermined: {defeater.detail}", Status.FACT, [defeater.citation])
    corrected = [e for e in inv.analysis.timeline if e.person == name and e.corrected_minutes]
    for entry in corrected:
        add(
            f"After clock correction, {entry.doc_id} puts {name} at {entry.location} at "
            f"{entry.time.split('T')[-1]}, not {entry.stated_time.split('T')[-1]} as stamped.",
            Status.FACT,
            [entry.citation],
            f"The stamp was adjusted by {entry.corrected_minutes} minutes.",
        )
    for extraction in inv.extractions.values():
        for assignment in extraction.credentials:
            if assignment.person == name:
                add(f"{assignment.credential} was assigned to {name}.", Status.FACT, [assignment.citation])
    for credential, reason in inv.analysis.compromised.items():
        related = [c for x in inv.analysis.contradictions if x.kind == "record_vs_record" for c in x.citations]
        add(f"{credential} was probably used by someone other than its holder. {reason}.", Status.ASSUMPTION, related)
    for contradiction in inv.analysis.contradictions:
        if contradiction.discredited_speaker == name:
            add(f"{name}'s account is contradicted: {contradiction.summary}", Status.FACT, contradiction.citations, contradiction.resolution)
    for link in leader.links:
        add(
            f"{name} is linked to {link.artifact} through: {' -> '.join(link.path)}.",
            Status.FACT,
            link.citations,
            "A shared identifier connects these records across documents.",
        )
    for signal in leader.motives[:2]:
        add(f"{name} may have had a motive: {signal.detail}", Status.ASSUMPTION, [signal.citation], "The stated circumstance is sourced; that it drove the crime is inference.")
    for signal in leader.means[:2]:
        add(f"{name} had the means: {signal.detail}", Status.ASSUMPTION, [signal.citation])
    for signal in leader.suspicious[:2]:
        add(f"Suspicious conduct by {name}: {signal.detail}", Status.ASSUMPTION, [signal.citation])
    for other in inv.rounds[-1].accomplices:
        suspect = next(s for s in inv.suspects if s.name == other)
        citations = [c for link in suspect.links for c in link.citations]
        add(f"{other} may have assisted {name}; the file shows a link but not what was agreed.", Status.SPECULATION, citations, key=False)
    for contradiction in inv.analysis.contradictions:
        speaker = contradiction.discredited_speaker
        if speaker and speaker != name:
            add(
                f"{speaker} gave a false account. The reason is unknown and need not relate to the crime.",
                Status.SPECULATION,
                contradiction.citations,
                key=False,
            )
    return support


def _elimination(suspect: SuspectAssessment) -> str:
    if suspect.opportunity == "excluded":
        return suspect.alibi_note or suspect.opportunity_note
    return f"Not excluded, but ranked lower (score {suspect.score}). Alibi: {suspect.alibi}. {suspect.alibi_note}".strip()


def _narrative(inv: Investigation, leader: SuspectAssessment | None) -> str:
    if leader is None:
        return "No suspect can be named."
    first = inv.rounds[0]
    parts = [f"The evidence points to {leader.name} ({leader.role})."]
    if leader.opportunity_note:
        parts.append(f"Opportunity: {leader.opportunity_note}.")
    if leader.alibi == "broken":
        parts.append(f"Their account does not survive: {leader.alibi_note}")
    if first.face_value_leader and first.face_value_leader != leader.name:
        parts.append(
            f"Taken at face value, the open file points to {first.face_value_leader}, but that reading does not "
            "survive source weighing, clock corrections and alibi checks."
        )
    return " ".join(parts)


def _profile(suspect: SuspectAssessment) -> str:
    return (
        f"{suspect.name}: opportunity {suspect.opportunity}, alibi {suspect.alibi}, score {suspect.score}. "
        f"{suspect.alibi_note or suspect.opportunity_note}".strip()
    )


def _on_topic(question: str, text: str) -> bool:
    """A passage must cover at least half of the question's content words."""
    wanted = set(tokenize(question))
    return len(wanted & set(tokenize(text))) * 2 >= len(wanted)


def _mentions(question: str, inv: Investigation, name: str) -> bool:
    person = next(p for p in inv.case.suspects if p.name == name)
    return any(re.search(rf"\b{re.escape(alias)}\b", question) for alias in [name, *person.aliases])
