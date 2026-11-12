"""Units of the deterministic layer: ingestion, retrieval, timeline, contradictions, joins."""
from __future__ import annotations

import pytest

from conftest import CASE_NAMES, CASES, new_investigation, solved
from detective.analysis import Geography, covers, overlaps, shift
from detective.heuristic import _resolve_weekday
from detective.ingest import load_case, parse_document, split_passages
from detective.models import Tier


def test_front_matter_is_required():
    with pytest.raises(ValueError):
        parse_document("no header here")


def test_log_documents_split_into_citable_lines():
    case = load_case(CASES / "case_01_missing_laptop")
    log = next(d for d in case.documents if d.id == "D2")
    passages = split_passages(log)
    assert passages[0].id == "D2:p1"
    assert any("19:42" in p.text and "19:58" not in p.text for p in passages)


def test_bar_tab_splits_by_line_despite_being_short():
    case = load_case(CASES / "case_02_aldermere_necklace")
    assert len(split_passages(next(d for d in case.documents if d.id == "A11"))) == 2


def test_sealed_documents_are_not_searchable_until_uncovered():
    inv = new_investigation("case_01_missing_laptop")
    assert not [p for p, _ in inv.store.search("QuickCash receipt") if p.doc_id == "S1"]
    inv.uncover("L1")
    assert any(p.doc_id == "S1" for p, _ in inv.store.search("QuickCash receipt"))


def test_retrieval_uses_aliases():
    inv = new_investigation("case_02_aldermere_necklace")
    hits = inv.store.search("Mr Hale salary", k=3)
    assert hits[0][0].doc_id == "A15"


def test_retrieval_is_ranked_and_repeatable():
    inv = new_investigation("case_03_halden_codex")
    first = inv.store.search("vault corridor camera offline")
    assert first == inv.store.search("vault corridor camera offline")
    assert [s for _, s in first] == sorted((s for _, s in first), reverse=True)


def test_alias_lookup_finds_every_mention():
    inv = new_investigation("case_03_halden_codex")
    documents = {p.doc_id for p in inv.store.passages_mentioning("Jonas Weber")}
    assert {"E3", "E4"} <= documents


def test_strict_overlap_ignores_touching_intervals():
    assert not overlaps("2025-01-01T10:00", "2025-01-01T11:00", "2025-01-01T11:00", "2025-01-01T12:00")
    assert overlaps("2025-01-01T10:00", "2025-01-01T11:30", "2025-01-01T11:00", "2025-01-01T12:00")
    assert covers("2025-01-01T10:00", "2025-01-01T11:00", "2025-01-01T10:30")


def test_shift_crosses_midnight():
    assert shift("2025-01-01T23:50", 25) == "2025-01-02T00:15"


def test_geography_containment():
    case = load_case(CASES / "case_02_aldermere_necklace")
    geo = Geography(case)
    assert geo.related("Room 512", "Fifth Floor")
    assert geo.related("Aldermere Hotel", "Room 512")
    assert not geo.related("Room 512", "Room 508")


def test_weekday_in_text_moves_the_date_forward():
    assert _resolve_weekday("2025-03-13", "on Friday morning") == "2025-03-14"
    assert _resolve_weekday("2025-03-13", "on Thursday evening") == "2025-03-13"


def test_timeline_is_chronological(case_name):
    inv = solved(case_name)
    times = [e.time for e in inv.analysis.timeline]
    assert times == sorted(times)


def test_clock_correction_keeps_the_stated_time():
    inv = solved("case_02_aldermere_necklace")
    shifted = [e for e in inv.analysis.timeline if e.corrected_minutes]
    assert shifted and all(e.doc_id == "A4" and e.corrected_minutes == -25 for e in shifted)
    assert shifted[0].stated_time != shifted[0].time


def test_a_clock_correction_leaves_other_devices_alone():
    inv = solved("case_02_aldermere_necklace")
    door = [e for e in inv.analysis.timeline if e.doc_id == "A2"]
    assert door and all(e.corrected_minutes == 0 for e in door)


def test_credential_events_are_attributed_only_after_the_register_is_opened():
    inv = new_investigation("case_02_aldermere_necklace")
    inv.reassess("start")
    door = next(e for e in inv.analysis.timeline if e.credential == "FOB-3")
    assert door.person is None
    inv.uncover("L2")
    door = next(e for e in inv.analysis.timeline if e.credential == "FOB-3")
    assert door.person == "Victor Hale" and door.attribution == "disputed", "the uncorrected footage still clashes"


def test_new_evidence_can_undo_an_earlier_inference():
    inv = new_investigation("case_02_aldermere_necklace")
    inv.reassess("start")
    inv.uncover("L2")
    assert "FOB-3" in inv.analysis.compromised
    inv.uncover("L1")
    door = next(e for e in inv.analysis.timeline if e.credential == "FOB-3")
    assert "FOB-3" not in inv.analysis.compromised
    assert door.attribution == "credential"
    assert inv.rounds[-1].leader == "Victor Hale"


def test_impossible_co_location_defeats_the_credential_not_the_person():
    inv = new_investigation("case_03_halden_codex")
    inv.reassess("start")
    inv.uncover("L1")
    clash = next(c for c in inv.analysis.contradictions if c.kind == "record_vs_record")
    assert clash.person == "Elias Brandt" and "BADGE-0007" in clash.resolution
    brandt = next(s for s in inv.suspects if s.name == "Elias Brandt")
    assert brandt.alibi == "verified" and brandt.opportunity == "excluded"


def test_cross_document_link_runs_through_a_shared_identifier():
    inv = solved("case_01_missing_laptop")
    link = next(l for l in inv.analysis.links if l.person == "Sam Whitlock")
    assert "+44 7700 900412" in link.path
    assert {"S1", "D11"} <= {c.passage_id.split(":")[0] for c in link.citations}


def test_hidden_connection_needs_the_registry_lead():
    inv = new_investigation("case_03_halden_codex")
    inv.reassess("start")
    assert not inv.analysis.links
    inv.uncover("L2")
    assert {l.person for l in inv.analysis.links} == {"Helena Marsh", "Jonas Weber"}


def test_source_tiers():
    inv = new_investigation("case_03_halden_codex")
    assert inv.store.documents["E2"].tier == Tier.RECORD
    assert inv.store.documents["E5"].tier == Tier.TESTIMONY
    assert inv.store.documents["E13"].tier == Tier.ANONYMOUS


@pytest.mark.parametrize("name", CASE_NAMES)
def test_offline_extraction_is_entirely_grounded(name):
    assert solved(name).rejections == []
