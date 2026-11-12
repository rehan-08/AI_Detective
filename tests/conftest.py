from __future__ import annotations

from pathlib import Path

import pytest

from detective.evaluation import load_solution
from detective.heuristic import HeuristicExtractor
from detective.ingest import load_case
from detective.investigation import Investigation
from detective.rules import RuleReasoner

CASES = Path(__file__).resolve().parent.parent / "cases"
CASE_NAMES = sorted(p.name for p in CASES.iterdir() if (p / "case.json").exists())


def new_investigation(name: str) -> Investigation:
    case = load_case(CASES / name)
    return Investigation(case, HeuristicExtractor(case))


def solved(name: str) -> Investigation:
    inv = new_investigation(name)
    RuleReasoner().investigate(inv)
    return inv


@pytest.fixture(params=CASE_NAMES)
def case_name(request) -> str:
    return request.param


@pytest.fixture
def solution(case_name):
    return load_solution(CASES / case_name)
