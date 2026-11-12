"""Transform the cases and check that conclusions move the way logic says they must.

These tests guard against a solver that is secretly tuned to specific names or files.
"""
from __future__ import annotations

from conftest import CASES
from detective.heuristic import HeuristicExtractor
from detective.ingest import load_case
from detective.investigation import Investigation
from detective.models import Case
from detective.rules import RuleReasoner


def swap_people(case: Case, first: str, second: str) -> Case:
    """Swap two suspects' identities in every document and in the case sheet."""
    raw = case.model_dump_json()
    pairs: list[tuple[str, str]] = []
    a = next(p for p in case.suspects if p.name == first)
    b = next(p for p in case.suspects if p.name == second)
    pairs.append((a.name, b.name))
    pairs.extend(zip(sorted(a.aliases, key=len, reverse=True), sorted(b.aliases, key=len, reverse=True)))
    for index, (left, right) in enumerate(pairs):
        raw = raw.replace(left, f"\0{index}L").replace(right, f"\0{index}R")
    for index, (left, right) in enumerate(pairs):
        raw = raw.replace(f"\\u0000{index}L", right).replace(f"\0{index}L", right)
        raw = raw.replace(f"\\u0000{index}R", left).replace(f"\0{index}R", left)
    return Case.model_validate_json(raw)


def run(case: Case) -> Investigation:
    inv = Investigation(case, HeuristicExtractor(case))
    RuleReasoner().investigate(inv)
    return inv


def test_swapping_culprit_and_decoy_swaps_the_verdict_case_one():
    case = load_case(CASES / "case_01_missing_laptop")
    swapped = swap_people(case, "Sam Whitlock", "Tom Reyes")
    assert run(swapped).conclusion.culprit == "Tom Reyes"


def test_swapping_culprit_and_decoy_swaps_the_verdict_case_two():
    case = load_case(CASES / "case_02_aldermere_necklace")
    swapped = swap_people(case, "Victor Hale", "Nadia Petrov")
    assert run(swapped).conclusion.culprit == "Nadia Petrov"


def test_swapping_culprit_and_decoy_swaps_the_verdict_case_three():
    case = load_case(CASES / "case_03_halden_codex")
    swapped = swap_people(case, "Helena Marsh", "Elias Brandt")
    assert run(swapped).conclusion.culprit == "Elias Brandt"


def test_removing_the_lie_removes_the_broken_alibi():
    case = load_case(CASES / "case_01_missing_laptop")
    case.documents = [d for d in case.documents if d.id != "D5"]
    inv = run(case)
    sam = next(s for s in inv.suspects if s.name == "Sam Whitlock")
    assert sam.alibi != "broken"


def test_removing_the_exculpatory_lead_leaves_the_framed_suspect_in_front():
    case = load_case(CASES / "case_03_halden_codex")
    case.leads = [lead for lead in case.leads if lead.id != "L1"]
    inv = run(case)
    brandt = next(s for s in inv.suspects if s.name == "Elias Brandt")
    assert brandt.alibi == "broken" and brandt.opportunity == "direct"
    assert inv.analysis.compromised == {}, "nothing in the file now shows the badge was misused"


def test_removing_the_clock_ticket_keeps_the_alibi_standing():
    case = load_case(CASES / "case_02_aldermere_necklace")
    case.leads = [lead for lead in case.leads if lead.id != "L1"]
    inv = run(case)
    hale = next(s for s in inv.suspects if s.name == "Victor Hale")
    assert hale.alibi == "verified", "without the ticket there is no basis to doubt the footage"
    assert not [e for e in inv.analysis.timeline if e.corrected_minutes]


def test_conclusions_are_deterministic():
    case = load_case(CASES / "case_03_halden_codex")
    first, second = run(case), run(case)
    assert first.rounds[-1].ranking == second.rounds[-1].ranking
    assert [f.statement for f in first.findings] == [f.statement for f in second.findings]
