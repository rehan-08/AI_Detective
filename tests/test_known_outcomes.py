"""End-to-end checks against each case's predefined solution, using the offline backend."""
from __future__ import annotations

import pytest

from conftest import CASE_NAMES, new_investigation, solved
from detective.evaluation import score
from detective.models import Confidence, Status
from detective.report import render_report
from detective.rules import RuleReasoner


def test_three_cases_cover_the_difficulty_levels():
    difficulties = {new_investigation(name).case.difficulty for name in CASE_NAMES}
    assert {"easy", "twisted", "impossible"} <= difficulties


def test_case_passes_its_answer_key(case_name, solution):
    result = score(solved(case_name), solution)
    assert result.passed, result.misses
    assert result.contradiction_recall == 1.0
    assert result.timeline_recall == 1.0
    assert result.retrieval_recall == 1.0
    assert result.key_evidence_recall == 1.0


def test_first_case_has_four_suspects():
    assert len(new_investigation("case_01_missing_laptop").case.suspects) == 4


def test_case_two_alibi_holds_until_the_clock_ticket_is_opened():
    inv = new_investigation("case_02_aldermere_necklace")
    inv.reassess("start")
    hale = next(s for s in inv.suspects if s.name == "Victor Hale")
    assert hale.alibi == "verified"
    assert inv.rounds[0].confidence == Confidence.LOW
    assert inv.rounds[0].surviving_suspects == []

    inv.uncover("L1")
    hale = next(s for s in inv.suspects if s.name == "Victor Hale")
    assert hale.alibi != "verified"
    corrected = [e for e in inv.analysis.timeline if e.doc_id == "A4"]
    assert corrected[0].time.endswith("22:05") and corrected[0].stated_time.endswith("22:30")


def test_case_three_flips_from_the_framed_suspect_when_leads_are_followed():
    inv = solved("case_03_halden_codex")
    assert inv.rounds[0].leader == "Elias Brandt"
    assert inv.rounds[0].confidence != Confidence.HIGH, "one badge record must not justify HIGH"
    assert inv.conclusion.culprit == "Helena Marsh"
    assert inv.conclusion.accomplices == ["Jonas Weber"]
    assert "BADGE-0007" in inv.analysis.compromised


def test_withholding_leads_keeps_confidence_honest():
    inv = new_investigation("case_03_halden_codex")
    conclusion = RuleReasoner().investigate(inv, auto_leads=False)
    assert conclusion.culprit == "Elias Brandt"
    assert conclusion.confidence == Confidence.MEDIUM
    assert inv.rounds[-1].open_questions


def test_anonymous_tip_with_injected_instructions_has_no_effect():
    inv = solved("case_03_halden_codex")
    assert "E13" not in inv.cited_documents()
    brandt = next(s for s in inv.suspects if s.name == "Elias Brandt")
    assert not any(s.citation.passage_id.startswith("E13") for s in brandt.motives + brandt.means)


def test_a_liar_with_a_verified_alibi_is_not_the_culprit():
    inv = solved("case_02_aldermere_necklace")
    clara = next(s for s in inv.suspects if s.name == "Clara Voss")
    assert clara.lies and clara.alibi == "verified" and clara.opportunity == "excluded"


def test_false_witness_is_discredited():
    inv = solved("case_02_aldermere_necklace")
    resolved = [c for c in inv.analysis.contradictions if c.discredited_speaker == "Marcus Bell"]
    assert resolved and set(resolved[0].documents) == {"A6", "A8"}


def test_report_separates_facts_assumptions_and_speculation(case_name):
    inv = solved(case_name)
    report = render_report(inv)
    assert "Most likely suspect" in report and "Confidence" in report
    for heading in ("## Facts", "## Assumptions", "## Speculation", "## Contradictions", "## Timeline"):
        assert heading in report
    statuses = {f.status for f in inv.findings}
    assert Status.FACT in statuses and Status.ASSUMPTION in statuses


def test_every_supporting_finding_is_cited_and_not_speculation(case_name):
    inv = solved(case_name)
    by_id = {f.id: f for f in inv.findings}
    assert inv.conclusion.supporting_findings
    for finding_id in inv.conclusion.supporting_findings:
        assert by_id[finding_id].citations
        assert by_id[finding_id].status != Status.SPECULATION


@pytest.mark.parametrize("name", CASE_NAMES)
def test_trace_records_a_reasoning_process(name):
    inv = solved(name)
    actions = [step.action for step in inv.steps]
    assert "extract" in actions and "assess" in actions and "finding" in actions
