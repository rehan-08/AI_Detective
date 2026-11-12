"""Offline extractor: a rule-based stand-in for the Claude extractor.

It needs no API key and is fully deterministic, which makes it useful for tests,
demos and as a baseline. It reads the actual text, but its rules were written
alongside the bundled cases, so it says nothing about how well the system handles
a new case. Claude is the extractor for real use.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from .ingest import split_passages
from .models import (
    Case,
    Citation,
    CredentialAssignment,
    Defeater,
    DocExtraction,
    Document,
    EntityMention,
    Event,
    Identifier,
    Passage,
    Signal,
)

TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
CREDENTIAL = re.compile(r"\b(?:BADGE|FOB|PIN)-\d+\b")
FULL_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
PHONE = re.compile(r"\+\d[\d ]{8,}\d")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
SERIAL = re.compile(r"\bserial(?: number)?:? ([A-Z0-9]{6,})\b")
ORGANISATION = re.compile(r"\b((?:[A-Z][a-z]+ )+(?:Ltd|LLC|Inc|GmbH))\b")
CLOCK = re.compile(
    r"(?:The )?(?P<device>[A-Z][\w ]+?) clock (?:runs|is|was) (?P<minutes>\d+) minutes (?P<dir>fast|ahead|slow|behind)"
)
COMPROMISED = re.compile(r"\b((?:BADGE|FOB|PIN)-\d+)\b[^.]*\b(lost|stolen|cloned|copied|shared|borrowed)\b", re.I)
SIGNED_OUT = re.compile(r"\b((?:BADGE|FOB|PIN)-\d+)\b (?:signed out to|assigned to|issued to)")
NEGATION = re.compile(r"\b(did not|didn't|never|was not|wasn't|not at)\b", re.I)
ABSENCE = re.compile(r"\b(seat is empty|absent|was missing from)\b", re.I)
TRANSIT = re.compile(r"\b(enters|leaves|arrives|exits)\b", re.I)
FIRST_PERSON = re.compile(r"\b(I|my|me|we|our)\b")
WITNESSED = re.compile(r"\bI (?:saw|noticed|spotted|watched)\b")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SIGNAL_RULES = {
    "motive": re.compile(
        r"\b(debts?|owes?|owe[sd]?|creditors?|salary advance|behind on rent|loans?|passed over|dismiss\w*|"
        r"rivals?|inherit|resent\w*|refused twice|called in|wanted this manuscript)\b",
        re.I,
    ),
    "means": re.compile(
        r"\b(master (?:key|code|override)|admin rights|trained in|forgery supplies|knows that vault)\b", re.I
    ),
    "suspicious": re.compile(
        r"\b(carrying (?:a|an) [a-z ]*(?:laptop|case|sleeve|package|box)|went offline|disabled|wiped|deleted|"
        r"no work order|restricted folios)\b",
        re.I,
    ),
}


class HeuristicExtractor:
    name = "heuristic"

    def __init__(self, case: Case):
        self.case = case
        self._people = _alias_patterns((p.name, p.aliases) for p in case.everyone())
        self._suspects = {p.name for p in case.suspects}
        self._places = sorted(
            ((alias, loc.name) for loc in case.locations for alias in {loc.name, *loc.aliases}),
            key=lambda item: -len(item[0]),
        )

    def extract(self, doc: Document) -> DocExtraction:
        passages = split_passages(doc)
        out = DocExtraction(
            doc_id=doc.id, entities=[], events=[], signals=[], defeaters=[], credentials=[], identifiers=[]
        )
        for passage in passages:
            self._entities(passage, out)
            self._identifiers(passage, out)
            self._defeaters(passage, out)
            self._signals(doc, passage, out)
        logged = [p for p in passages if "|" in p.text]
        out.events.extend(self._log_events(doc, logged))
        for passage in passages:
            if "|" not in passage.text:
                out.events.extend(self._prose_events(doc, passage))
        return out

    # --- entities and joins -------------------------------------------------

    def _entities(self, passage: Passage, out: DocExtraction) -> None:
        for name in self._find_people(passage.text):
            out.entities.append(EntityMention(name=name, kind="person", citation=self._cite(passage, name)))
        place = self._find_place(passage.text)
        if place:
            out.entities.append(EntityMention(name=place[1], kind="location", citation=self._cite(passage, place[0])))

    def _identifiers(self, passage: Passage, out: DocExtraction) -> None:
        people = self._find_people(passage.text)
        holder = people[0] if len(people) == 1 else (people[0] if people else None)
        found: list[tuple[str, str]] = []
        found += [(m, "phone") for m in PHONE.findall(passage.text)]
        found += [(m, "email") for m in EMAIL.findall(passage.text)]
        found += [(m, "serial") for m in SERIAL.findall(passage.text)]
        found += [(m, "organisation") for m in ORGANISATION.findall(passage.text)]
        for value in self.case.crime.artifact_ids:
            if value in passage.text and not any(value == f[0] for f in found):
                found.append((value, "other"))
        for value, kind in found:
            out.identifiers.append(
                Identifier(value=value, kind=kind, linked_person=holder, citation=Citation(passage_id=passage.id, quote=value))
            )

    def _defeaters(self, passage: Passage, out: DocExtraction) -> None:
        for match in CLOCK.finditer(passage.text):
            sign = 1 if match["dir"] in ("fast", "ahead") else -1
            out.defeaters.append(
                Defeater(
                    kind="clock_offset",
                    target=match["device"].strip(),
                    minutes=sign * int(match["minutes"]),
                    detail=match.group(0),
                    citation=Citation(passage_id=passage.id, quote=match.group(0)),
                )
            )
        for match in COMPROMISED.finditer(passage.text):
            out.defeaters.append(
                Defeater(
                    kind="credential_compromised",
                    target=match.group(1),
                    minutes=None,
                    detail=match.group(0),
                    citation=Citation(passage_id=passage.id, quote=match.group(0)),
                )
            )
        for match in SIGNED_OUT.finditer(passage.text):
            people = self._find_people(passage.text)
            if people:
                out.credentials.append(
                    CredentialAssignment(
                        credential=match.group(1), person=people[0], citation=Citation(passage_id=passage.id, quote=match.group(1))
                    )
                )

    def _signals(self, doc: Document, passage: Passage, out: DocExtraction) -> None:
        if doc.type == "anonymous_tip":
            return
        for sentence in _sentences(passage.text):
            subjects = self._signal_subjects(doc, sentence)
            for kind, rule in SIGNAL_RULES.items():
                match = rule.search(sentence)
                if not match:
                    continue
                for person in subjects:
                    out.signals.append(
                        Signal(person=person, kind=kind, detail=sentence, citation=Citation(passage_id=passage.id, quote=sentence))
                    )

    def _signal_subjects(self, doc: Document, sentence: str) -> list[str]:
        chat = re.match(r".*?\|.*?\|\s*([^:|]+?)(?: to [^:|]+)?:", sentence)
        if chat:
            author = self._find_people(chat.group(1))
            return [p for p in author if p in self._suspects]
        named = [p for p in self._find_people(sentence) if p in self._suspects]
        if named:
            return named[:1]
        if doc.speaker in self._suspects and FIRST_PERSON.search(sentence):
            return [doc.speaker]
        return []

    # --- presence events ----------------------------------------------------

    def _log_events(self, doc: Document, passages: list[Passage]) -> list[Event]:
        events: list[Event] = []
        open_entries: dict[tuple, tuple[str, Passage, str]] = {}
        sightings: dict[tuple, tuple[str, str, Passage]] = {}

        def flush(key: tuple) -> None:
            if key in sightings:
                first, last, passage = sightings.pop(key)
                person, place = key
                events.append(self._event(person, None, place, first, None if last == first else last, passage, True))

        for passage in passages:
            fields = [f.strip() for f in passage.text.split("|")]
            clock = TIME.search(fields[0])
            if not clock:
                continue
            stamp = self._stamp(doc, clock, fields[0])
            people = self._find_people(passage.text) if doc.type != "chat" else []
            credential = CREDENTIAL.search(passage.text)
            place = self._find_place(passage.text)
            direction = fields[-1].upper()
            if direction in ("ENTRY", "EXIT") and place and (people or credential):
                person = people[0] if people else None
                cred = credential.group(0) if credential else None
                key = (person, cred, place[1])
                if direction == "ENTRY":
                    open_entries[key] = (stamp, passage, place[1])
                elif key in open_entries:
                    start, opening, _ = open_entries.pop(key)
                    events.append(self._event(person, cred, place[1], start, stamp, opening, True))
                else:
                    events.append(self._event(person, cred, place[1], stamp, None, passage, True))
            elif place and doc.type == "cctv":
                place = self._find_place(fields[1]) or place  # where the camera is, not where someone is heading
                for clause in re.split(r"(?<=\.)\s+", fields[-1]):
                    absent = bool(ABSENCE.search(clause))
                    for person in self._find_people(clause):
                        key = (person, place[1])
                        if absent:
                            flush(key)
                            events.append(self._event(person, None, place[1], stamp, None, passage, False))
                        elif TRANSIT.search(clause):
                            flush(key)
                            events.append(self._event(person, None, place[1], stamp, None, passage, True))
                        elif key in sightings:
                            sightings[key] = (sightings[key][0], stamp, sightings[key][2])
                        else:
                            sightings[key] = (stamp, stamp, passage)
        for key in list(sightings):
            flush(key)
        for (person, cred, place), (start, passage, _) in open_entries.items():
            events.append(self._event(person, cred, place, start, None, passage, True))
        return events

    def _prose_events(self, doc: Document, passage: Passage) -> list[Event]:
        events = []
        for sentence in _sentences(passage.text):
            if sentence.startswith("Q:") or "?" in sentence:
                continue
            text = sentence.removeprefix("A:").strip()
            times = list(TIME.finditer(text))
            if not times:
                continue
            person = self._subject(doc, text)
            if not person:
                continue
            start = self._stamp(doc, times[0], text)
            end = self._stamp(doc, times[1], text) if len(times) > 1 else None
            approximate = bool(re.search(r"\b(about|around|roughly)\b", text))
            if re.search(r"\b(?:went|gone) (?:straight )?home\b", text):
                place = f"Home of {person}"
                if re.search(r"\b(all evening|all night)\b", text):
                    end = f"{doc.event_date}T23:59"
                events.append(self._prose_event(person, place, start, end, passage, text, True, approximate))
                continue
            found = self._find_place(text)
            if not found:
                continue
            if re.search(r"\bfrom\s+\d", text) and end is None:
                end = self.case.crime.window_end
            if "until" in text and re.search(r"\buntil\s+\d", text) and not re.search(r"\b(from|between)\s+\d", text):
                start, end = self._stamp(doc, times[0], text), None
            present = not NEGATION.search(text)
            events.append(self._prose_event(person, found[1], start, end, passage, text, present, approximate))
        return events

    def _subject(self, doc: Document, text: str) -> str | None:
        people = self._find_people(text)
        if WITNESSED.search(text) and people:
            return people[0]
        if re.search(r"\bI\b", text) and doc.speaker:
            return doc.speaker
        return people[0] if people else None

    def _prose_event(self, person, place, start, end, passage, text, present, approximate) -> Event:
        return Event(
            description=text,
            person=person,
            credential=None,
            location=place,
            start=start,
            end=end,
            present=present,
            is_presence=True,
            approximate=approximate,
            citation=Citation(passage_id=passage.id, quote=text),
        )

    def _event(self, person, credential, place, start, end, passage, present) -> Event:
        who = person or credential
        action = "at" if present else "not at"
        return Event(
            description=f"{who} {action} {place}",
            person=person,
            credential=credential,
            location=place,
            start=start,
            end=end,
            present=present,
            is_presence=True,
            approximate=False,
            citation=Citation(passage_id=passage.id, quote=passage.text),
        )

    # --- helpers ------------------------------------------------------------

    def _stamp(self, doc: Document, clock: re.Match, context: str) -> str:
        explicit = FULL_DATE.search(context)
        day = explicit.group(1) if explicit else _resolve_weekday(doc.event_date, context)
        return f"{day}T{int(clock.group(1)):02d}:{clock.group(2)}"

    def _find_people(self, text: str) -> list[str]:
        hits = []
        for name, pattern in self._people:
            match = pattern.search(text)
            if match:
                hits.append((match.start(), name))
        return [name for _, name in sorted(hits)]

    def _find_place(self, text: str) -> tuple[str, str] | None:
        lowered = text.lower()
        for alias, name in self._places:
            if re.search(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", lowered):
                return alias, name
        return None

    def _cite(self, passage: Passage, needle: str) -> Citation:
        match = re.search(re.escape(needle), passage.text, re.I)
        return Citation(passage_id=passage.id, quote=match.group(0) if match else passage.text)


def _alias_patterns(entries) -> list[tuple[str, re.Pattern]]:
    patterns = []
    for name, aliases in entries:
        names = sorted({name, *aliases}, key=len, reverse=True)
        patterns.append((name, re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w-])")))
    return patterns


def _sentences(text: str) -> list[str]:
    parts = (s.strip().removeprefix("A:").strip() for s in re.split(r"(?<=[.?!])\s+|\n+", text))
    return [s for s in parts if s]


def _resolve_weekday(event_date: str, text: str) -> str:
    """Move to the following matching weekday when the text names a different day."""
    base = date.fromisoformat(event_date)
    for index, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}\b", text) and index != base.weekday():
            return (base + timedelta(days=(index - base.weekday()) % 7)).isoformat()
    return event_date
