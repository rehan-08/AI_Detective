"""A case written after the offline rules were frozen, in wording they were not tuned for.

The rules were built alongside the three bundled cases, so those scores flatter them.
This case is the honest measure. It uses plain speech ("knocked off about five",
"half past six") and unfamiliar log formats on purpose. Do not tune the rules to pass
it: that would destroy the only unbiased signal the offline backend has.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from detective.heuristic import HeuristicExtractor
from detective.ingest import load_case
from detective.investigation import Investigation
from detective.models import Confidence
from detective.rules import RuleReasoner

FOLDER = Path(__file__).parent / "fixtures" / "heldout_orchid"


@pytest.fixture
def investigation() -> Investigation:
    case = load_case(FOLDER)
    inv = Investigation(case, HeuristicExtractor(case))
    RuleReasoner().investigate(inv)
    return inv


def test_case_is_well_formed_and_runs_to_a_conclusion(investigation):
    assert investigation.conclusion is not None


def test_whatever_the_rules_miss_they_do_not_invent(investigation):
    assert investigation.rejections == []
    assert all(investigation.grounding.check_citation(c)[0] for c in investigation.all_citations())


def test_unfamiliar_wording_does_not_produce_false_confidence(investigation):
    solution = json.loads((FOLDER / "solution.json").read_text(encoding="utf-8"))
    if investigation.conclusion.culprit != solution["culprit"]:
        assert investigation.conclusion.confidence != Confidence.HIGH


@pytest.mark.xfail(strict=False, reason="Known limit: offline rules are not expected to generalise to new wording")
def test_offline_rules_find_the_culprit_on_unseen_wording(investigation):
    solution = json.loads((FOLDER / "solution.json").read_text(encoding="utf-8"))
    assert investigation.conclusion.culprit == solution["culprit"]
