"""Follow-up questions, reassessment and the boundary around the answer key."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from conftest import new_investigation, solved
from detective.rules import RuleReasoner

ROOT = Path(__file__).resolve().parent.parent


def ask(name: str, question: str):
    return RuleReasoner().answer(solved(name), question)


def test_followup_about_a_suspect_returns_cited_profile():
    answer = ask("case_02_aldermere_necklace", "What is the alibi of Clara Voss?")
    assert "Clara Voss" in answer.answer and answer.citations


def test_followup_citations_verify():
    inv = solved("case_03_halden_codex")
    answer = RuleReasoner().answer(inv, "Who disabled the vault corridor camera?")
    assert answer.citations
    assert all(inv.grounding.check_citation(c)[0] for c in answer.citations)


def test_question_with_no_evidence_gets_no_invented_answer():
    answer = ask("case_01_missing_laptop", "What did the submarine crew report?")
    assert not answer.grounded and not answer.citations
    assert "no evidence" in answer.answer


def test_question_about_a_real_detail_is_answered():
    answer = ask("case_02_aldermere_necklace", "Who signed out FOB-3?")
    assert any("FOB-3 signed out to Victor Hale" in c.quote for c in answer.citations)


def test_uncovering_reassesses_and_records_the_change():
    inv = new_investigation("case_03_halden_codex")
    inv.reassess("start")
    before = inv.rounds[-1].leader
    inv.uncover("L1", "Testing the alibi")
    assert before == "Elias Brandt" and inv.rounds[-1].leader == "Helena Marsh"
    assert any("leader changed" in step.observation for step in inv.steps)


def test_uncovering_twice_is_refused():
    inv = new_investigation("case_01_missing_laptop")
    inv.uncover("L1")
    try:
        inv.uncover("L1")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_only_the_evaluator_reads_the_answer_key():
    readers = [
        path.name
        for path in (ROOT / "detective").glob("*.py")
        if "solution.json" in path.read_text(encoding="utf-8")
    ]
    assert readers == ["evaluation.py"]


def cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    command = [sys.executable, "-m", "detective", *args]
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env)


def test_cli_evaluate_runs_offline_and_passes():
    result = cli("evaluate", "--backend", "offline")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS") == 3


def test_cli_session_uncovers_a_lead_then_answers_from_it(tmp_path):
    name = "case_03_halden_codex"
    first = cli("solve", name, "--no-leads", cwd=tmp_path)
    assert "Elias Brandt" in first.stdout, first.stderr

    second = cli("uncover", name, "L1", cwd=tmp_path)
    assert "Before: Elias Brandt" in second.stdout and "After:  Helena Marsh" in second.stdout, second.stderr

    third = cli("ask", name, "Who is on the Brandt landline call to Greta Brandt?", cwd=tmp_path)
    assert "ES1" in third.stdout, third.stderr

    report = (tmp_path / "runs" / name / "report.md").read_text(encoding="utf-8")
    assert "## Evidence that was undermined" in report and "BADGE-0007" in report
    assert "Sources given no weight" in report and "E13" in report


def test_cli_rejects_an_unknown_lead(tmp_path):
    result = cli("uncover", "case_01_missing_laptop", "L9", cwd=tmp_path)
    assert result.returncode != 0 and "not a pending lead" in result.stderr
