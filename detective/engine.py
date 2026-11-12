"""Suspect scoring: motive, means, opportunity and alibi as explicit rules.

The weights are deliberately simple and live in one place. They rank suspects and
set a ceiling on what the evidence can justify; they are not probabilities.
"""
from __future__ import annotations

from .analysis import Analysis, Geography, minutes_between, overlaps
from .models import (
    Assessment,
    Case,
    Citation,
    Confidence,
    Signal,
    SuspectAssessment,
    Tier,
    TimelineEntry,
)
from .store import EvidenceStore

WEIGHTS = {
    "opportunity_direct": 3.0,
    "opportunity_site": 2.0,
    "excluded": -4.0,
    "alibi_broken": 2.5,
    "motive": 1.0,
    "means": 1.0,
    "suspicious": 1.0,
    "linked_to_artifact": 3.0,
    "unrelated_lie": 0.5,
}
MAX_SIGNALS = 2
ALIBI_COVERAGE = 0.8  # share of the crime window a record must cover to clear someone
MIN_OPPORTUNITY_MINUTES = 5  # shorter gaps are not a realistic chance to commit the act
HIGH_MARGIN, MEDIUM_MARGIN = 3.0, 1.5


class Engine:
    def __init__(self, case: Case, store: EvidenceStore, analysis: Analysis):
        self.case = case
        self.store = store
        self.analysis = analysis
        self.geo = Geography(case)
        self.window = case.crime.window_start, case.crime.window_end
        self.crime_location = case.crime.location

    def assess_all(self) -> list[SuspectAssessment]:
        results = [self._assess(p.name, p.role) for p in self.case.suspects]
        return sorted(results, key=lambda s: (-s.score, s.name))

    def face_value_leader(self) -> str | None:
        """Who an evidence count would blame: no alibis, defeaters or source weighing."""
        scores: dict[str, float] = {}
        for suspect in self.case.suspects:
            score = 0.0
            for entry in self._entries(suspect.name):
                if entry.present and overlaps(entry.time, entry.end, *self.window):
                    level = self._level(entry)
                    if level:
                        score = max(score, WEIGHTS[f"opportunity_{level}"])
            for kind in ("motive", "means"):
                score += min(len(self._signals(suspect.name, kind)), MAX_SIGNALS)
            scores[suspect.name] = score
        return max(sorted(scores), key=lambda n: scores[n]) if scores else None

    # --- one suspect --------------------------------------------------------

    def _assess(self, name: str, role: str) -> SuspectAssessment:
        result = SuspectAssessment(name=name, role=role)
        result.motives = self._signals(name, "motive")
        result.means = self._signals(name, "means")
        result.suspicious = self._signals(name, "suspicious")
        result.links = [link for link in self.analysis.links if link.person == name]
        self._opportunity(result)
        self._alibi(result)
        self._score(result)
        return result

    def _entries(self, name: str) -> list[TimelineEntry]:
        return [e for e in self.analysis.timeline if e.person == name and e.is_presence and e.location]

    def _admissible(self, name: str) -> list[tuple[int, TimelineEntry]]:
        """Positive presence entries that survived contradiction checks."""
        return [
            (i, e)
            for i, e in enumerate(self.analysis.timeline)
            if e.person == name and e.is_presence and e.location and e.present
            and i not in self.analysis.refuted
            and e.attribution != "disputed"
            and e.credential not in self.analysis.compromised
        ]

    def _level(self, entry: TimelineEntry) -> str | None:
        location = entry.location
        if location == self.crime_location or self.crime_location in self.geo.ancestors(location):
            return "direct"
        if location in self.geo.ancestors(self.crime_location):
            return "site"
        return None

    def _opportunity(self, result: SuspectAssessment) -> None:
        best, evidence = None, None
        for _, entry in self._admissible(result.name):
            if entry.tier < Tier.DOCUMENT and entry.speaker == result.name:
                continue  # placing yourself at a scene is not corroboration of opportunity
            if not self._touches_window(entry):
                continue
            level = self._level(entry)
            if level == "direct" or (level == "site" and best is None):
                best, evidence = level, entry
        if best:
            result.opportunity = best
            result.opportunity_note = (
                f"{evidence.description} ({_clock(evidence.time)}"
                f"{'-' + _clock(evidence.end) if evidence.end else ''}, {evidence.doc_id})"
            )
            result.support.append(evidence.citation)
        site = self._complete_site()
        if site and not self._had_time_on_site(result.name, site):
            result.opportunity = "excluded"
            result.opportunity_note = (
                f"{site} has a complete access log, and it shows no usable time on site inside the crime window."
            )

    def _touches_window(self, entry: TimelineEntry) -> bool:
        start, end = self.window
        return overlaps(entry.time, entry.end, start, end) or start <= entry.time <= end

    def _complete_site(self) -> str | None:
        logged = {loc.name for loc in self.case.locations if loc.complete_access_log}
        return next((p for p in self.geo.ancestors(self.crime_location) if p in logged), None)

    def _had_time_on_site(self, name: str, site: str) -> bool:
        start, end = self.window
        on_site = [
            (max(e.time, start), min(e.end, end))
            for _, e in self._admissible(name)
            if e.tier >= Tier.DOCUMENT and e.end and e.location == site and e.time < end and e.end > start
        ]
        elsewhere = self._elsewhere_intervals(name)
        for span_start, span_end in on_site:
            free = minutes_between(span_start, span_end)
            for other_start, other_end in elsewhere:
                lo, hi = max(span_start, other_start), min(span_end, other_end)
                if lo < hi:
                    free -= minutes_between(lo, hi)
            if free >= MIN_OPPORTUNITY_MINUTES:
                return True
        return False

    def _elsewhere_intervals(self, name: str) -> list[tuple[str, str]]:
        """Record-backed intervals at places that do not contain the crime scene."""
        return [
            (e.time, e.end)
            for _, e in self._admissible(name)
            if e.tier >= Tier.RECORD and e.end and not self.geo.related(e.location, self.crime_location)
        ]

    def _alibi(self, result: SuspectAssessment) -> None:
        start, end = self.window
        total = minutes_between(start, end)
        covered = 0.0
        alibi_doc = None
        for other_start, other_end in self._elsewhere_intervals(result.name):
            lo, hi = max(other_start, start), min(other_end, end)
            if lo < hi:
                covered += minutes_between(lo, hi)
        if total and covered / total >= ALIBI_COVERAGE:
            alibi_doc = next(
                e for _, e in self._admissible(result.name)
                if e.tier >= Tier.RECORD and e.end and not self.geo.related(e.location, self.crime_location)
                and overlaps(e.time, e.end, start, end)
            )
        lies = [
            c for c in self.analysis.contradictions
            if c.discredited_speaker == result.name
        ]
        result.lies = [c.id for c in lies]
        window_lies = [c for c in lies if c.overlap_start <= end and c.overlap_end >= start]
        if alibi_doc:
            result.alibi = "verified"
            result.alibi_note = f"{alibi_doc.description} ({alibi_doc.doc_id}) covers the crime window."
            result.opportunity = "excluded"
            result.opportunity_note = "A record places this person elsewhere during the crime window."
            result.support.append(alibi_doc.citation)
        elif window_lies:
            result.alibi = "broken"
            result.alibi_note = window_lies[0].summary
            result.support.extend(window_lies[0].citations)
        elif any(e.speaker == result.name and not self.geo.related(e.location, self.crime_location) for e in self._entries(result.name)):
            result.alibi = "unverified"
            result.alibi_note = "Rests on this person's own word, with no independent record."

    def _score(self, result: SuspectAssessment) -> None:
        parts: dict[str, float] = {}
        if result.opportunity == "excluded":
            parts["excluded"] = WEIGHTS["excluded"]
        elif result.opportunity in ("direct", "site"):
            parts[f"opportunity_{result.opportunity}"] = WEIGHTS[f"opportunity_{result.opportunity}"]
        if result.alibi == "broken":
            parts["alibi_broken"] = WEIGHTS["alibi_broken"]
        for kind, items in (("motive", result.motives), ("means", result.means), ("suspicious", result.suspicious)):
            if items:
                parts[kind] = WEIGHTS[kind] * min(len(items), MAX_SIGNALS)
        if result.links:
            parts["linked_to_artifact"] = WEIGHTS["linked_to_artifact"]
        if result.lies and result.alibi != "broken":
            parts["unrelated_lie"] = WEIGHTS["unrelated_lie"]
        result.breakdown = parts
        result.score = round(sum(parts.values()), 2)

    def _signals(self, name: str, kind: str) -> list[Signal]:
        seen, unique = set(), []
        for signal in self.analysis.signals:
            if signal.person != name or signal.kind != kind:
                continue
            tier = self.store.documents[signal.citation.passage_id.split(":")[0]].tier
            if tier == Tier.ANONYMOUS or signal.citation.quote in seen:
                continue
            seen.add(signal.citation.quote)
            unique.append(signal)
        return unique

    # --- the round-level judgement ------------------------------------------

    def judge(self, ranked: list[SuspectAssessment], round_no: int, trigger: str) -> Assessment:
        surviving = [s for s in ranked if s.opportunity != "excluded"]
        pool = surviving or ranked
        leader = pool[0] if pool else None
        runner_score = pool[1].score if len(pool) > 1 else (ranked[1].score if len(ranked) > 1 else 0.0)
        margin = leader.score - runner_score if leader else 0.0
        records = self._record_support(leader) if leader else set()
        level, why = self._confidence(leader, margin, records, bool(surviving))
        return Assessment(
            round=round_no,
            trigger=trigger,
            leader=leader.name if leader else None,
            confidence=level,
            confidence_explanation=why,
            ranking=[(s.name, s.score) for s in ranked],
            face_value_leader=self.face_value_leader(),
            surviving_suspects=[s.name for s in surviving],
            open_questions=self._open_questions(leader, surviving, records),
            accomplices=[s.name for s in surviving if leader and s.name != leader.name and s.links],
        )

    def _record_support(self, suspect: SuspectAssessment) -> set[str]:
        docs = {
            e.doc_id for _, e in self._admissible(suspect.name)
            if e.tier >= Tier.RECORD and self._touches_window(e)
        }
        docs |= {c.passage_id.split(":")[0] for link in suspect.links for c in link.citations}
        return {d for d in docs if self.store.documents[d].tier >= Tier.RECORD}

    def _confidence(self, leader, margin, records, anyone_left) -> tuple[Confidence, str]:
        if not leader:
            return Confidence.LOW, "There are no suspects to assess."
        if not anyone_left:
            return Confidence.LOW, (
                "Every suspect currently has a verified alibi, so at least one record must be wrong. "
                f"{leader.name} is named only as the least excluded suspect."
            )
        facts = (
            f"margin over the next suspect {margin:.1f}, opportunity {leader.opportunity}, alibi {leader.alibi}, "
            f"{len(records)} independent record source(s): {', '.join(sorted(records)) or 'none'}"
        )
        strong_tie = leader.opportunity == "direct" or bool(leader.links)
        if margin >= HIGH_MARGIN and len(records) >= 2 and leader.opportunity in ("direct", "site") and strong_tie:
            return Confidence.HIGH, f"Independent records converge on this suspect ({facts})."
        if margin >= MEDIUM_MARGIN:
            return Confidence.MEDIUM, f"The suspect stands out, but the case rests on too few independent records ({facts})."
        return Confidence.LOW, f"The evidence does not separate this suspect from the others ({facts})."

    def _open_questions(self, leader, surviving, records) -> list[str]:
        questions = []
        if not surviving:
            questions.append("Every alibi holds, so which record is flawed? Check camera clocks and device timestamps.")
        if leader and leader.opportunity in ("unknown", "excluded"):
            questions.append(f"What places {leader.name} near {self.crime_location}? Check lift, door and key records.")
        if leader and len(records) < 2:
            questions.append(
                f"Do independent records support the case against {leader.name}? Test any badge or credential "
                "use against the alibi, and check call records."
            )
        if leader and not leader.links and self.case.crime.artifact_ids:
            questions.append(
                f"Who is behind {', '.join(self.case.crime.artifact_ids)}? Follow company, phone and bank records."
            )
        return questions


def _clock(iso: str) -> str:
    return iso.split("T")[-1]


def dedupe_citations(citations: list[Citation]) -> list[Citation]:
    seen, unique = set(), []
    for citation in citations:
        key = (citation.passage_id, citation.quote)
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique
