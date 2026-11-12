"""Structured representation of a case and everything derived from it.

Every derived item carries a Citation, so a conclusion can always be traced to a
verbatim span of a source document. Times are ISO strings ("2025-03-13T19:42"),
which sort correctly as text and survive JSON round trips.
"""
from __future__ import annotations

from enum import Enum, IntEnum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Tier(IntEnum):
    """How much weight a source deserves, independent of what it says."""

    ANONYMOUS = 0
    TESTIMONY = 1
    DOCUMENT = 2
    RECORD = 3


DOC_TIERS: dict[str, Tier] = {
    "anonymous_tip": Tier.ANONYMOUS,
    "witness_interview": Tier.TESTIMONY,
    "incident_report": Tier.TESTIMONY,
    "chat": Tier.DOCUMENT,
    "email": Tier.DOCUMENT,
    "document": Tier.DOCUMENT,
    "system_log": Tier.RECORD,
    "cctv": Tier.RECORD,
    "financial_record": Tier.RECORD,
    "phone_record": Tier.RECORD,
}


class Status(str, Enum):
    """Epistemic status of a statement in the report."""

    FACT = "fact"
    ASSUMPTION = "assumption"
    SPECULATION = "speculation"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Document(BaseModel):
    id: str
    type: str
    title: str
    date: str
    event_date: str
    speaker: Optional[str] = None
    device: Optional[str] = None
    text: str

    @property
    def tier(self) -> Tier:
        return DOC_TIERS.get(self.type, Tier.DOCUMENT)


class Passage(BaseModel):
    id: str  # "D2:p4"
    doc_id: str
    index: int
    text: str


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    quote: str


class Person(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = ""


class Location(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    within: Optional[str] = None
    complete_access_log: bool = False


class Lead(BaseModel):
    id: str
    title: str
    document: str


class Crime(BaseModel):
    summary: str
    location: str
    window_start: str
    window_end: str
    artifact_ids: list[str] = Field(default_factory=list)


class Case(BaseModel):
    id: str
    title: str
    difficulty: str
    briefing: str
    crime: Crime
    suspects: list[Person]
    people: list[Person] = Field(default_factory=list)
    locations: list[Location]
    leads: list[Lead] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    sealed: dict[str, Document] = Field(default_factory=dict)  # lead id -> document

    def everyone(self) -> list[Person]:
        return self.suspects + self.people


# --- Extraction schema -------------------------------------------------------
# These models are the contract between an extractor (Claude or the offline
# heuristics) and the deterministic analysis. All fields are required so the same
# models can be used as a strict JSON schema for structured outputs.


class Event(BaseModel):
    """Something that happened, or that someone says happened."""

    model_config = ConfigDict(extra="forbid")

    description: str
    person: Optional[str] = Field(description="Canonical person name, or null when only a credential is known")
    credential: Optional[str] = Field(description="Badge, fob or PIN identifier the source attributes this to, or null")
    location: Optional[str] = Field(description="Canonical location name, or null")
    start: Optional[str] = Field(description="YYYY-MM-DDTHH:MM exactly as stated, before any clock correction")
    end: Optional[str] = Field(description="YYYY-MM-DDTHH:MM or null for a point in time")
    present: bool = Field(description="False when the source says the person was NOT at the location")
    is_presence: bool = Field(description="True when this places a person at a location over the given time")
    approximate: bool
    citation: Citation


class Signal(BaseModel):
    """Evidence bearing on motive, means or suspicious behaviour of one person."""

    model_config = ConfigDict(extra="forbid")

    person: str
    kind: Literal["motive", "means", "suspicious"]
    detail: str
    citation: Citation


class Defeater(BaseModel):
    """A statement that undermines another piece of evidence."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["clock_offset", "credential_compromised", "record_unreliable"]
    target: str = Field(description="Device name for clock/record problems; credential ID for compromised credentials")
    minutes: Optional[int] = Field(description="Minutes the clock runs fast (positive) or slow (negative); null otherwise")
    detail: str
    citation: Citation


class CredentialAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credential: str
    person: str
    citation: Citation


class Identifier(BaseModel):
    """A phone number, serial, company name or similar that can join documents."""

    model_config = ConfigDict(extra="forbid")

    value: str
    kind: Literal["phone", "email", "serial", "organisation", "account", "other"]
    linked_person: Optional[str] = Field(description="Person this passage ties the identifier to, or null")
    citation: Citation


class EntityMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["person", "location", "object", "organisation"]
    citation: Citation


class DocExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    entities: list[EntityMention]
    events: list[Event]
    signals: list[Signal]
    defeaters: list[Defeater]
    credentials: list[CredentialAssignment]
    identifiers: list[Identifier]


class ExtractionRejection(BaseModel):
    """An extracted item the grounding guard refused to admit."""

    doc_id: str
    item_type: str
    reason: str
    content: str


# --- Analysis output ---------------------------------------------------------


class TimelineEntry(BaseModel):
    time: str
    end: Optional[str] = None
    stated_time: str
    person: Optional[str]
    credential: Optional[str] = None
    location: Optional[str]
    description: str
    tier: Tier
    doc_id: str
    speaker: Optional[str] = None
    present: bool = True
    is_presence: bool = True
    corrected_minutes: int = 0
    attribution: Literal["direct", "credential", "disputed"] = "direct"
    citation: Citation


class Contradiction(BaseModel):
    id: str
    person: str
    summary: str
    documents: list[str]
    citations: list[Citation]
    kind: Literal["claim_vs_record", "claim_vs_claim", "record_vs_record", "semantic"]
    resolution: str
    discredited_speaker: Optional[str] = None
    overlap_start: str = ""
    overlap_end: str = ""


class Link(BaseModel):
    """A chain of shared identifiers tying a person to a crime artefact."""

    person: str
    artifact: str
    path: list[str]
    citations: list[Citation]


class SuspectAssessment(BaseModel):
    name: str
    role: str = ""
    score: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    opportunity: Literal["direct", "site", "unknown", "excluded"] = "unknown"
    opportunity_note: str = ""
    alibi: Literal["verified", "broken", "unverified", "none"] = "none"
    alibi_note: str = ""
    motives: list[Signal] = Field(default_factory=list)
    means: list[Signal] = Field(default_factory=list)
    suspicious: list[Signal] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    lies: list[str] = Field(default_factory=list)  # contradiction ids
    support: list[Citation] = Field(default_factory=list)


class Finding(BaseModel):
    id: str = ""
    statement: str
    status: Status
    citations: list[Citation] = Field(default_factory=list)
    rationale: str = ""


class ReasoningStep(BaseModel):
    round: int
    action: str
    observation: str
    citations: list[Citation] = Field(default_factory=list)


class Assessment(BaseModel):
    """The state of the investigation at the end of one round."""

    round: int
    trigger: str
    leader: Optional[str]
    confidence: Confidence
    confidence_explanation: str
    ranking: list[tuple[str, float]]
    face_value_leader: Optional[str] = None
    surviving_suspects: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    accomplices: list[str] = Field(default_factory=list)


class Conclusion(BaseModel):
    culprit: Optional[str]
    confidence: Confidence
    confidence_explanation: str
    narrative: str
    accomplices: list[str] = Field(default_factory=list)
    supporting_findings: list[str] = Field(default_factory=list)
    eliminated: dict[str, str] = Field(default_factory=dict)
    engine_agrees: bool = True
    engine_note: str = ""


class Answer(BaseModel):
    question: str
    answer: str
    status: Status
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
