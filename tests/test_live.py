"""Runs the real system with a real model. Skipped unless credentials are available.

    pytest -m live

Each case costs real tokens: one extraction call per document plus an agent loop.
"""
from __future__ import annotations

import os

import pytest

from conftest import CASE_NAMES, CASES
from detective.evaluation import load_solution, score
from detective.ingest import load_case
from detective.investigation import Investigation

live = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")


@pytest.mark.live
@live
@pytest.mark.parametrize("name", CASE_NAMES)
def test_claude_solves_the_case(name):
    from detective.claude_backend import ClaudeExtractor, ClaudeGateway, ClaudeReasoner

    case = load_case(CASES / name)
    gateway = ClaudeGateway()
    inv = Investigation(case, ClaudeExtractor(case, gateway))
    ClaudeReasoner(gateway).investigate(inv)
    result = score(inv, load_solution(CASES / name))
    assert result.culprit_correct, result.misses
    assert result.citation_validity == 1.0
    assert result.red_herrings_clean, result.misses


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY is not set")
@pytest.mark.parametrize("name", CASE_NAMES)
def test_groq_solves_the_case(name):
    from detective.groq_backend import GroqExtractor, GroqGateway, GroqReasoner

    case = load_case(CASES / name)
    gateway = GroqGateway()
    inv = Investigation(case, GroqExtractor(case, gateway))
    GroqReasoner(gateway).investigate(inv)
    result = score(inv, load_solution(CASES / name))
    assert result.culprit_correct, result.misses
    assert result.citation_validity == 1.0
    assert result.red_herrings_clean, result.misses
