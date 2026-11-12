"""The anti-hallucination gate must reject anything the source does not say."""
from __future__ import annotations

from conftest import new_investigation
from detective.grounding import normalize, time_in_text
from detective.models import Citation, DocExtraction, Event, Finding, Signal, Status

CASE = "case_01_missing_laptop"


def event(**overrides) -> Event:
    values = dict(
        description="x",
        person="Sam Whitlock",
        credential=None,
        location="Brightwell Studio",
        start="2025-03-13T19:42",
        end=None,
        present=True,
        is_presence=True,
        approximate=False,
        citation=Citation(passage_id="D2:p11", quote="19:42 | Sam Whitlock"),
    )
    return Event(**(values | overrides))


def filtered(inv, **fields) -> tuple[DocExtraction, list]:
    base = dict(doc_id="D2", entities=[], events=[], signals=[], defeaters=[], credentials=[], identifiers=[])
    return inv.grounding.filter_extraction(DocExtraction(**(base | fields)))


def test_verbatim_quote_passes():
    inv = new_investigation(CASE)
    citation, reason = inv.grounding.check_citation(Citation(passage_id="D5:p1", quote="I left the studio at 18:05"))
    assert citation and not reason


def test_fabricated_quote_is_rejected():
    inv = new_investigation(CASE)
    citation, reason = inv.grounding.check_citation(Citation(passage_id="D5:p1", quote="I stole the laptop"))
    assert citation is None and "not found" in reason


def test_nonexistent_document_is_rejected():
    inv = new_investigation(CASE)
    citation, reason = inv.grounding.check_citation(Citation(passage_id="Z9:p1", quote="anything at all"))
    assert citation is None and "no such source" in reason


def test_wrong_passage_is_repaired_within_the_same_document():
    inv = new_investigation(CASE)
    citation, _ = inv.grounding.check_citation(Citation(passage_id="D5:p9", quote="I did not come back"))
    assert citation.passage_id == "D5:p2"


def test_sealed_evidence_cannot_be_cited_before_it_is_uncovered():
    inv = new_investigation(CASE)
    assert inv.grounding.check_citation(Citation(passage_id="S1:p1", quote="Receipt 88213"))[0] is None
    inv.uncover("L1")
    assert inv.grounding.check_citation(Citation(passage_id="S1:p1", quote="Receipt 88213"))[0] is not None


def test_invented_timestamp_is_rejected():
    inv = new_investigation(CASE)
    passage = next(p for p in inv.store.doc_passages("D2") if "19:42" in p.text)
    good = event(citation=Citation(passage_id=passage.id, quote="Sam Whitlock"))
    bad = event(start="2025-03-13T03:15", citation=Citation(passage_id=passage.id, quote="Sam Whitlock"))
    kept, rejected = filtered(inv, events=[good, bad])
    assert len(kept.events) == 1 and len(rejected) == 1
    assert "does not appear" in rejected[0].reason


def test_person_absent_from_the_passage_is_rejected():
    inv = new_investigation(CASE)
    passage = next(p for p in inv.store.doc_passages("D2") if "19:42" in p.text)
    framed = event(person="Tom Reyes", citation=Citation(passage_id=passage.id, quote="19:42"))
    kept, rejected = filtered(inv, events=[framed])
    assert not kept.events and "not mentioned" in rejected[0].reason


def test_unknown_person_and_location_are_rejected():
    inv = new_investigation(CASE)
    passage = next(p for p in inv.store.doc_passages("D2") if "19:42" in p.text)
    citation = Citation(passage_id=passage.id, quote="19:42")
    kept, rejected = filtered(
        inv,
        events=[event(person="Mystery Man", citation=citation), event(location="Secret Bunker", citation=citation)],
    )
    assert not kept.events and len(rejected) == 2


def test_signal_with_a_fake_quote_is_rejected():
    inv = new_investigation(CASE)
    fake = Signal(person="Tom Reyes", kind="motive", detail="x", citation=Citation(passage_id="D3:p1", quote="I wanted revenge"))
    kept, rejected = filtered(inv, signals=[fake])
    assert not kept.signals and rejected


def test_first_person_statement_can_cite_a_passage_without_the_name():
    inv = new_investigation(CASE)
    passage = "D5:p1"
    speaker_event = event(
        location="Home of Sam Whitlock",
        start="2025-03-13T18:05",
        citation=Citation(passage_id=passage, quote="I left the studio at 18:05"),
    )
    kept, rejected = filtered(inv, doc_id="D5", events=[speaker_event])
    assert kept.events and not rejected


def test_fact_without_citation_is_rejected():
    inv = new_investigation(CASE)
    finding, reason = inv.add_finding(Finding(statement="Sam did it", status=Status.FACT))
    assert finding is None and "must cite" in reason
    assert inv.finding_rejections


def test_fact_resting_on_one_interested_statement_is_downgraded():
    inv = new_investigation(CASE)
    claim = Finding(
        statement="Sam went home",
        status=Status.FACT,
        citations=[Citation(passage_id="D5:p1", quote="went straight home")],
    )
    finding, _ = inv.add_finding(claim)
    assert finding.status == Status.ASSUMPTION
    assert "Downgraded" in finding.rationale


def test_fact_from_a_record_stays_a_fact():
    inv = new_investigation(CASE)
    finding, _ = inv.add_finding(
        Finding(statement="Sam re-entered", status=Status.FACT, citations=[Citation(passage_id="D8:p2", quote="Sam Whitlock enters")])
    )
    assert finding.status == Status.FACT


def test_speculation_may_stand_without_citation():
    inv = new_investigation(CASE)
    finding, _ = inv.add_finding(Finding(statement="Perhaps a buyer was waiting", status=Status.SPECULATION))
    assert finding is not None


def test_normalisation_handles_curly_quotes_and_whitespace():
    assert normalize("Sam’s   \n bag") == normalize("sam's bag")


def test_time_formats():
    assert time_in_text("2025-03-13T21:05", "at about 21:05 we spoke")
    assert time_in_text("2025-03-13T21:00", "around 9 pm")
    assert not time_in_text("2025-03-13T21:05", "at about 22:05")
