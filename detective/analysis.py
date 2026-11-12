"""Deterministic analysis: timeline, contradictions and identifier links.

This layer never asks a model anything. It takes verified extractions and applies
explicit rules, so a change in a conclusion can always be traced to a rule and a
cited source. Each round re-runs it from scratch, so new evidence can undo an
earlier inference rather than merely add to it.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations

from .grounding import normalize
from .models import (
    Case,
    Citation,
    Contradiction,
    Defeater,
    DocExtraction,
    Event,
    Link,
    Signal,
    Tier,
    TimelineEntry,
)
from .store import EvidenceStore


class Geography:
    """Containment between locations: a room is inside a floor, inside a building."""

    def __init__(self, case: Case):
        self._parent = {loc.name: loc.within for loc in case.locations}

    def ancestors(self, name: str) -> list[str]:
        chain, current = [], self._parent.get(name)
        while current:
            chain.append(current)
            current = self._parent.get(current)
        return chain

    def related(self, a: str, b: str) -> bool:
        return a == b or a in self.ancestors(b) or b in self.ancestors(a)


def overlaps(a_start: str, a_end: str | None, b_start: str, b_end: str | None) -> bool:
    """Strict overlap, so a departure at 18:05 does not clash with arriving at 18:05."""
    return a_start < (b_end or b_start) and b_start < (a_end or a_start)


def covers(start: str, end: str | None, point: str) -> bool:
    return start <= point <= (end or start)


def shift(iso: str, minutes: int) -> str:
    return (datetime.fromisoformat(iso) + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M")


def minutes_between(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60


@dataclass
class Analysis:
    timeline: list[TimelineEntry]
    contradictions: list[Contradiction]
    signals: list[Signal]
    defeaters: list[Defeater]
    links: list[Link]
    credential_owner: dict[str, str]
    compromised: dict[str, str] = field(default_factory=dict)
    refuted: set[int] = field(default_factory=set)  # indices into timeline


def analyse(case: Case, store: EvidenceStore, extractions: list[DocExtraction], extra: list[Contradiction] | None = None) -> Analysis:
    geo = Geography(case)
    defeaters = [d for ex in extractions for d in ex.defeaters]
    signals = [s for ex in extractions for s in ex.signals]
    owner = {c.credential: c.person for ex in extractions for c in ex.credentials}
    compromised = {d.target: d.detail for d in defeaters if d.kind == "credential_compromised"}

    timeline = _build_timeline(store, extractions, defeaters, owner)
    contradictions, refuted = _detect(timeline, geo, compromised, store)
    for entry in timeline:
        if entry.credential in compromised and entry.attribution == "credential":
            entry.attribution = "disputed"
    contradictions.extend(extra or [])
    return Analysis(
        timeline=timeline,
        contradictions=contradictions,
        signals=signals,
        defeaters=defeaters,
        links=_find_links(case, extractions),
        credential_owner=owner,
        compromised=compromised,
        refuted=refuted,
    )


def _build_timeline(store, extractions, defeaters, owner) -> list[TimelineEntry]:
    offsets = _clock_offsets(store, defeaters)
    entries = []
    for extraction in extractions:
        doc = store.documents[extraction.doc_id]
        offset = offsets.get(doc.id, 0)
        for event in extraction.events:
            if not event.start:
                continue
            entries.append(_entry(event, doc, offset, owner))
    return sorted(entries, key=lambda e: (e.time, e.doc_id))


def _clock_offsets(store: EvidenceStore, defeaters: list[Defeater]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for defeater in defeaters:
        if defeater.kind != "clock_offset" or not defeater.minutes:
            continue
        target = normalize(defeater.target)
        for doc in store.documents.values():
            device = normalize(doc.device or "")
            if device and (target in device or device in target):
                offsets[doc.id] = defeater.minutes
    return offsets


def _entry(event: Event, doc, offset: int, owner: dict[str, str]) -> TimelineEntry:
    person = event.person or owner.get(event.credential or "")
    return TimelineEntry(
        time=shift(event.start, -offset),
        end=shift(event.end, -offset) if event.end else None,
        stated_time=event.start,
        person=person,
        credential=event.credential,
        location=event.location,
        description=event.description,
        tier=doc.tier,
        doc_id=doc.id,
        speaker=doc.speaker,
        present=event.present,
        is_presence=event.is_presence,
        corrected_minutes=-offset,
        attribution="credential" if event.credential else "direct",
        citation=event.citation,
    )


def _conflict(a: TimelineEntry, b: TimelineEntry, geo: Geography) -> bool:
    if not (a.is_presence and b.is_presence and a.location and b.location):
        return False
    if a.doc_id == b.doc_id or not overlaps(a.time, a.end, b.time, b.end):
        return False
    if a.present and b.present:
        return not geo.related(a.location, b.location)
    if a.present != b.present:
        return geo.related(a.location, b.location)
    return False


def _detect(timeline, geo, compromised, store) -> tuple[list[Contradiction], set[int]]:
    """Find clashes per person. Clashes between records are settled first, because
    they can show that a credential was misused, which dissolves any clash that
    depended on that credential."""
    by_person: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(timeline):
        if entry.person and entry.is_presence:
            by_person[entry.person].append(index)

    pairs: list[tuple[str, int, int]] = []
    for person, indices in by_person.items():
        for i, j in combinations(indices, 2):
            if _conflict(timeline[i], timeline[j], geo):
                pairs.append((person, i, j))

    found: dict[tuple, Contradiction] = {}
    refuted: set[int] = set()
    for person, i, j in pairs:
        a, b = timeline[i], timeline[j]
        if a.tier == Tier.RECORD and b.tier == Tier.RECORD:
            weak = next((e for e in (a, b) if e.attribution == "credential"), None)
            other = b if weak is a else a
            if weak and other.attribution == "direct":
                compromised.setdefault(
                    weak.credential,
                    f"{weak.credential} recorded at {weak.location} while {person} is independently recorded at {other.location}",
                )
                refuted.add(i if weak is a else j)
            _record(found, person, a, b, "record_vs_record", _settle_records(person, weak, other), None)

    testimony_only: list[tuple[str, int, int]] = []
    for person, i, j in pairs:
        a, b = timeline[i], timeline[j]
        if a.tier == Tier.RECORD and b.tier == Tier.RECORD:
            continue
        if any(e.credential in compromised for e in (a, b)):
            continue
        strong, weak = (a, b) if a.tier >= b.tier else (b, a)
        if strong.tier >= Tier.DOCUMENT:
            refuted.add(j if strong is a else i)
            speaker = weak.speaker or weak.person
            note = (
                f"The {_label(store, strong)} ({strong.doc_id}) outweighs the statement in {weak.doc_id}; "
                f"{speaker}'s account is unreliable on this point."
            )
            _record(found, person, a, b, "claim_vs_record", note, speaker)
        else:
            testimony_only.append((person, i, j))

    for person, i, j in testimony_only:
        a, b = timeline[i], timeline[j]
        loser = next((e for index, e in ((i, a), (j, b)) if index in refuted), None)
        note = f"Settled by a record that refutes {loser.doc_id}." if loser else "Unresolved: both sources are testimony."
        _record(found, person, a, b, "claim_vs_claim", note, None)
    return list(found.values()), refuted


def _settle_records(person, weak, other) -> str:
    if weak and other.attribution == "direct":
        return (
            f"{person} cannot be in two places at once. The direct record ({other.doc_id}) stands, so the use of "
            f"{weak.credential} in {weak.doc_id} is not attributed to {person}."
        )
    return "Two records conflict and neither can be preferred; check both devices."


def _label(store, entry) -> str:
    return store.documents[entry.doc_id].title


def _record(found, person, a, b, kind, resolution, discredited) -> None:
    key = (person, *sorted((a.doc_id, b.doc_id)))
    start, end = max(a.time, b.time), min(a.end or a.time, b.end or b.time)
    if key in found:
        existing = found[key]
        existing.overlap_start = min(existing.overlap_start, start)
        existing.overlap_end = max(existing.overlap_end, end)
        return
    state_a = "at" if a.present else "not at"
    state_b = "at" if b.present else "not at"
    summary = (
        f"{a.doc_id} places {person} {state_a} {a.location} ({_span(a)}), "
        f"but {b.doc_id} places {person} {state_b} {b.location} ({_span(b)})."
    )
    found[key] = Contradiction(
        id=f"C-{person.split()[-1]}-{a.doc_id}-{b.doc_id}",
        person=person,
        summary=summary,
        documents=[a.doc_id, b.doc_id],
        citations=[a.citation, b.citation],
        kind=kind,
        resolution=resolution,
        discredited_speaker=discredited,
        overlap_start=start,
        overlap_end=end,
    )


def _span(entry: TimelineEntry) -> str:
    clock = entry.time.split("T")[-1]
    return f"{clock}-{entry.end.split('T')[-1]}" if entry.end else clock


def _find_links(case: Case, extractions: list[DocExtraction]) -> list[Link]:
    """Breadth-first search over identifiers that co-occur in a passage."""
    graph: dict[str, set[str]] = defaultdict(set)
    edge_citation: dict[frozenset, Citation] = {}
    by_passage: dict[str, list] = defaultdict(list)
    suspects = {p.name for p in case.suspects}

    def connect(a: str, b: str, citation: Citation) -> None:
        graph[a].add(b)
        graph[b].add(a)
        edge_citation.setdefault(frozenset((a, b)), citation)

    mentions: dict[str, list[Citation]] = defaultdict(list)
    for extraction in extractions:
        for identifier in extraction.identifiers:
            by_passage[identifier.citation.passage_id].append(identifier)
            mentions[normalize(identifier.value)].append(identifier.citation)
            if identifier.linked_person in suspects:
                connect(normalize(identifier.value), identifier.linked_person, identifier.citation)
    for items in by_passage.values():
        for first, second in combinations(items, 2):
            connect(normalize(first.value), normalize(second.value), first.citation)

    links = []
    for artifact in case.crime.artifact_ids:
        start = normalize(artifact)
        previous: dict[str, str | None] = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in sorted(graph[node]):
                if neighbour in previous:
                    continue
                previous[neighbour] = node
                if neighbour not in suspects:
                    queue.append(neighbour)
        for person in sorted(suspects & previous.keys()):
            path = [person]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            citations = [edge_citation[frozenset(pair)] for pair in zip(path, path[1:])] + mentions[start]
            links.append(Link(person=person, artifact=artifact, path=path, citations=citations))
    return links
