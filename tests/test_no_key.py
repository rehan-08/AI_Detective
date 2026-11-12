"""Everything that can be proved about the Claude path without an API key.

This mirrors tests/test_live.py with a fake API, then covers what a real model or
service may do wrong. These tests show the plumbing and guards work. They cannot show
that Claude reasons well: DetectiveModel simply files the rule engine's leader.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import anthropic
import pytest
from fastapi.testclient import TestClient

import detective.claude_backend as backend
from conftest import CASE_NAMES, CASES
from fake_claude import DetectiveModel, FakeApi, api_error, message, scripted, text, tool_use
from detective.claude_backend import ClaudeExtractor, ClaudeReasoner
from detective.evaluation import load_solution, score
from detective.heuristic import HeuristicExtractor
from detective.ingest import load_case
from detective.investigation import Investigation
from detective.models import Citation, DocExtraction, Signal, Status
from detective.server import create_app
from detective.workspace import Backends, claude_configured

ROOT = Path(__file__).resolve().parent.parent
LAPTOP, NECKLACE, CODEX = "case_01_missing_laptop", "case_02_aldermere_necklace", "case_03_halden_codex"
HEADERS = {"x-detective-client": "web"}
KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")


@pytest.fixture
def fake_api():
    created = []

    def make(respond):
        api = FakeApi(respond)
        created.append(api)
        return api

    yield make
    for api in created:
        api.close()


def claude_investigation(case, api, cache_dir=None):
    gateway = api.gateway()
    return Investigation(case, ClaudeExtractor(case, gateway, cache_dir=cache_dir)), ClaudeReasoner(gateway)


def app_client(tmp_path, api) -> TestClient:
    def factory(case):
        gateway = api.gateway()
        return ClaudeExtractor(case, gateway, cache_dir=None), ClaudeReasoner(gateway)

    backends = Backends(claude_factory=factory, claude_available=lambda: True)
    return TestClient(create_app(data_dir=tmp_path, backends=backends), headers=HEADERS)


def run(client, case_id, auto_leads=True) -> dict:
    response = client.post(f"/api/cases/{case_id}/runs", json={"backend": "claude", "auto_leads": auto_leads})
    assert response.status_code == 202, response.text
    client.app.state.manager.get(case_id).wait()
    return client.get(f"/api/cases/{case_id}/state").json()


def empty_extraction(doc_id="D1") -> str:
    return DocExtraction(doc_id=doc_id, entities=[], events=[], signals=[], defeaters=[], credentials=[], identifiers=[]).model_dump_json()


def extraction_then(agent):
    """Answer extraction requests with nothing, and agent turns with `agent`."""

    def respond(request):
        if "format" in request.get("output_config", {}):
            return text(empty_extraction())
        return agent(request)

    return respond


def error_blocks(request) -> list[str]:
    content = request["messages"][-1]["content"]
    return [b["content"] for b in content if isinstance(b, dict) and b.get("is_error")]


# --- the keyless mirror of tests/test_live.py -----------------------------------


@pytest.mark.parametrize("name", CASE_NAMES)
def test_claude_path_completes_each_case_with_no_key(name, fake_api):
    case = load_case(CASES / name)
    api = fake_api(DetectiveModel(case))
    inv, reasoner = claude_investigation(case, api)
    reasoner.investigate(inv)

    result = score(inv, load_solution(CASES / name))
    assert result.culprit_correct, result.misses
    assert result.citation_validity == 1.0
    assert result.red_herrings_clean, result.misses
    assert result.contradiction_recall == 1.0 and result.timeline_recall == 1.0
    assert result.confidence_ok and result.accomplices_correct
    assert inv.uncovered == [lead.id for lead in case.leads]
    assert not inv.rejections and not inv.finding_rejections


def test_every_request_is_well_formed(fake_api):
    case = load_case(CASES / CODEX)
    api = fake_api(DetectiveModel(case))
    inv, reasoner = claude_investigation(case, api)
    reasoner.investigate(inv)

    expected = len(case.documents) + len(case.leads)
    extraction = [r for r in api.requests if "format" in r.get("output_config", {})]
    turns = [r for r in api.requests if "tools" in r]
    assert len(extraction) == expected and len(turns) >= 5
    for request in api.requests:
        assert request["model"] == "claude-opus-5-5" and request["fallbacks"] == "default"
        assert not {"temperature", "top_p", "top_k"} & request.keys()
    assert all(r["messages"][0]["role"] == "user" for r in turns)
    assert all("server-side-fallback-2026-07-01" in h["anthropic-beta"] for h in api.headers)


def test_documents_reach_the_model_as_delimited_data(fake_api):
    case = load_case(CASES / CODEX)
    api = fake_api(DetectiveModel(case))
    claude_investigation(case, api)
    tip = next(r["messages"][0]["content"] for r in api.requests if "doc_id: E13" in r["messages"][0]["content"])
    assert '<passage id="E13:p2">SYSTEM NOTICE' in tip
    assert "untrusted evidence" in api.requests[0]["system"]


def test_sealed_text_never_reaches_the_model_before_it_is_opened(fake_api):
    case = load_case(CASES / CODEX)
    api = fake_api(DetectiveModel(case))
    inv, reasoner = claude_investigation(case, api)
    reasoner.investigate(inv, auto_leads=False)
    sent = json.dumps(api.requests)
    assert "H. Aldous" not in sent and "PIN-4471" not in sent
    assert "uncover_lead" not in {t["name"] for r in api.requests if "tools" in r for t in r["tools"]}


def test_extraction_cache_avoids_paying_twice(fake_api, tmp_path):
    case = load_case(CASES / LAPTOP)
    api = fake_api(DetectiveModel(case))
    claude_investigation(case, api, cache_dir=tmp_path)
    first = len(api.requests)
    claude_investigation(load_case(CASES / LAPTOP), api, cache_dir=tmp_path)
    assert first == len(case.documents) and len(api.requests) == first


# --- the application on the Claude path ------------------------------------------


def test_app_journey_on_the_claude_path(fake_api, tmp_path):
    case = load_case(CASES / CODEX)
    api = fake_api(DetectiveModel(case))
    client = app_client(tmp_path, api)

    state = run(client, CODEX, auto_leads=False)
    assert state["status"] == "done", state["error"]
    assert state["backend"] == "claude" and state["conclusion"]["culprit"] == "Elias Brandt"

    for lead in ("L1", "L2", "L3"):
        assert client.post(f"/api/cases/{CODEX}/leads/{lead}/uncover").status_code == 202
        client.app.state.manager.get(CODEX).wait()
    state = client.get(f"/api/cases/{CODEX}/state").json()
    assert state["status"] == "done", state["error"]
    assert state["conclusion"]["culprit"] == "Helena Marsh"
    assert client.get(f"/api/cases/{CODEX}/evaluation").json()["culprit_correct"] is True
    assert "Helena Marsh" in client.get(f"/api/cases/{CODEX}/report.md").text


def test_restart_makes_no_model_calls(fake_api, tmp_path):
    case = load_case(CASES / LAPTOP)
    api = fake_api(DetectiveModel(case))
    run(app_client(tmp_path, api), LAPTOP)
    calls = len(api.requests)

    reopened = app_client(tmp_path, api)
    state = reopened.get(f"/api/cases/{LAPTOP}/state").json()
    assert state["backend"] == "claude" and state["conclusion"]["culprit"] == "Sam Whitlock"
    assert reopened.get(f"/api/cases/{LAPTOP}/report.md").status_code == 200
    assert len(api.requests) == calls


# --- a model that misbehaves ------------------------------------------------------


def good_finding():
    return {
        "statement": "Sam came back.",
        "status": "fact",
        "citations": [{"passage_id": "D8:p2", "quote": "Sam Whitlock enters"}],
        "rationale": "CCTV.",
    }


def conclusion(culprit="Sam Whitlock", finding="F1"):
    case = load_case(CASES / LAPTOP)
    return {
        "culprit": culprit,
        "accomplices": [],
        "confidence": "HIGH",
        "confidence_explanation": "Stated.",
        "narrative": "Stated.",
        "supporting_finding_ids": [finding],
        "eliminated": [{"name": p.name, "reason": "Lower."} for p in case.suspects if p.name != culprit],
        "disagreement_with_engine": "The scripted extraction is empty.",
    }


def test_model_obeying_an_injected_note_cannot_close_the_case(fake_api):
    """A fact resting only on the anonymous note is downgraded, so it cannot carry a verdict."""
    case = load_case(CASES / CODEX)
    obedient = {
        "statement": "Elias Brandt took the Codex.",
        "status": "fact",
        "citations": [{"passage_id": "E13:p1", "quote": "Elias Brandt took the Codex"}],
        "rationale": "The notice says the case is closed.",
    }
    names = [p.name for p in case.suspects if p.name != "Elias Brandt"]
    verdict = {
        "culprit": "Elias Brandt",
        "accomplices": [],
        "confidence": "HIGH",
        "confidence_explanation": "As instructed.",
        "narrative": "As instructed.",
        "supporting_finding_ids": ["F1"],
        "eliminated": [{"name": n, "reason": "Ignored."} for n in names],
        "disagreement_with_engine": "",
    }
    api = fake_api(extraction_then(scripted([
        message([tool_use(1, "record_finding", obedient)], "tool_use"),
        message([tool_use(2, "submit_conclusion", verdict)], "tool_use"),
        api_error(500),
    ])))
    inv, reasoner = claude_investigation(case, api)
    with pytest.raises(anthropic.InternalServerError):
        reasoner.investigate(inv)

    assert inv.findings[0].status == Status.ASSUMPTION
    assert "Downgraded" in inv.findings[0].rationale
    assert inv.conclusion is None
    assert any("at least one supporting fact" in e for e in error_blocks(api.requests[-1]))


def test_signal_extracted_from_an_anonymous_source_carries_no_weight():
    """The offline extractor skips anonymous tips, but Claude's extraction might not."""
    case = load_case(CASES / CODEX)
    inv = Investigation(case, HeuristicExtractor(case))
    inv.reassess("start")
    before = next(s for s in inv.suspects if s.name == "Elias Brandt").score
    rumour = Signal(
        person="Elias Brandt",
        kind="suspicious",
        detail="He was seen at the Vault",
        citation=Citation(passage_id="E13:p1", quote="He was seen at the Vault at 21:15"),
    )
    inv.extractions["E13"].signals.append(rumour)
    inv.reassess("with the rumour")
    brandt = next(s for s in inv.suspects if s.name == "Elias Brandt")
    assert not brandt.suspicious and brandt.score == before


def test_model_citing_sealed_evidence_is_rejected(fake_api):
    case = load_case(CASES / LAPTOP)
    peek = good_finding() | {"citations": [{"passage_id": "S1:p1", "quote": "Receipt 88213"}]}
    api = fake_api(extraction_then(scripted([
        message([tool_use(1, "record_finding", peek)], "tool_use"),
        message([tool_use(2, "record_finding", good_finding())], "tool_use"),
        message([tool_use(3, "submit_conclusion", conclusion())], "tool_use"),
    ])))
    inv, reasoner = claude_investigation(case, api)
    reasoner.investigate(inv)
    assert len(inv.findings) == 1 and len(inv.finding_rejections) == 1


def test_extractor_that_invents_everything_admits_nothing(fake_api):
    case = load_case(CASES / LAPTOP)
    invented = {
        "doc_id": "D1",
        "entities": [],
        "events": [{
            "description": "A confession",
            "person": "Sam Whitlock",
            "credential": None,
            "location": "Studio B",
            "start": "2025-03-13T19:45",
            "end": None,
            "present": True,
            "is_presence": True,
            "approximate": False,
            "citation": {"passage_id": "D1:p1", "quote": "Sam confessed to everything"},
        }],
        "signals": [{"person": "Sam Whitlock", "kind": "motive", "detail": "Greed", "citation": {"passage_id": "D1:p1", "quote": "Sam is greedy"}}],
        "defeaters": [{"kind": "clock_offset", "target": "Lobby camera", "minutes": 90, "detail": "x", "citation": {"passage_id": "D1:p1", "quote": "the clock is wrong"}}],
        "credentials": [],
        "identifiers": [],
    }
    api = fake_api(lambda _request: text(json.dumps(invented)))
    inv = Investigation(case, ClaudeExtractor(case, api.gateway(), cache_dir=None))
    inv.reassess("start")
    assert not inv.analysis.timeline and not inv.analysis.signals and not inv.analysis.defeaters
    assert len(inv.rejections) == 3 * len(case.documents)


def test_tool_call_with_wrong_arguments_is_returned_as_an_error(fake_api):
    case = load_case(CASES / LAPTOP)
    api = fake_api(extraction_then(scripted([
        message([tool_use(1, "search_evidence", {"wrong": "field"}), tool_use(2, "no_such_tool", {})], "tool_use"),
        message([tool_use(3, "record_finding", good_finding())], "tool_use"),
        message([tool_use(4, "submit_conclusion", conclusion())], "tool_use"),
    ])))
    inv, reasoner = claude_investigation(case, api)
    reasoner.investigate(inv)
    errors =[e for request in api.requests for e in error_blocks(request)]
    assert any("invalid arguments" in e for e in errors) and any("unknown tool" in e for e in errors)
    assert inv.conclusion.culprit == "Sam Whitlock"


def test_model_that_never_concludes_ends_in_a_visible_failure(fake_api, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "MAX_TURNS", 3)
    api = fake_api(extraction_then(lambda _request: text("Still thinking.")))
    state = run(app_client(tmp_path, api), LAPTOP)
    assert state["status"] == "failed" and "no conclusion after 3 turns" in state["error"]


def test_malformed_structured_output_ends_in_a_visible_failure(fake_api, tmp_path):
    api = fake_api(lambda _request: text("this is not json"))
    state = run(app_client(tmp_path, api), LAPTOP)
    assert state["status"] == "failed" and state["error"]


# --- the service failing ----------------------------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [
        (api_error(401, "authentication_error"), "rejected the API key"),
        (api_error(429, "rate_limit_error"), "rate limit"),
        (api_error(500), "error (500)"),
        (api_error(529, "overloaded_error"), "error (529)"),
        (message([], "max_tokens"), "truncated"),
        (message([], "refusal") | {"stop_details": {"type": "refusal", "category": None, "explanation": "Not this one."}}, "declined this request. Details: Not this one."),
    ],
)
def test_failures_during_a_run_are_explained(fake_api, tmp_path, reply, expected):
    api = fake_api(lambda _request: reply)
    state = run(app_client(tmp_path, api), LAPTOP)
    assert state["status"] == "failed"
    assert expected in state["error"]


def test_a_failing_run_stops_calling_the_api(fake_api, tmp_path):
    """After a bad key or a rate limit, the remaining documents must not be attempted."""
    api = fake_api(lambda _request: api_error(401, "authentication_error"))
    state = run(app_client(tmp_path, api), LAPTOP)
    assert state["status"] == "failed"
    assert len(api.requests) < len(load_case(CASES / LAPTOP).documents), "only calls already in flight may finish"


def test_unreachable_service_is_explained(fake_api, tmp_path):
    api = fake_api(lambda _request: text("unused"))
    client = app_client(tmp_path, api)
    api.close()
    state = run(client, LAPTOP)
    assert state["status"] == "failed" and "reach the Claude API" in state["error"]


def test_a_failed_run_keeps_the_previous_result_on_disk(fake_api, tmp_path):
    case = load_case(CASES / LAPTOP)
    api = fake_api(DetectiveModel(case))
    client = app_client(tmp_path, api)
    assert run(client, LAPTOP)["status"] == "done"

    broken = fake_api(lambda _request: api_error(500))
    assert run(app_client(tmp_path, broken), LAPTOP)["status"] == "failed"
    reopened = TestClient(create_app(data_dir=tmp_path), headers=HEADERS)
    assert reopened.get(f"/api/cases/{LAPTOP}/state").json()["conclusion"]["culprit"] == "Sam Whitlock"


def test_question_failure_is_explained(fake_api, tmp_path):
    case = load_case(CASES / LAPTOP)
    model = DetectiveModel(case)

    def respond(request):
        if "follow-up question" in str(request.get("system")):
            return api_error(429, "rate_limit_error")
        return model(request)

    client = app_client(tmp_path, fake_api(respond))
    run(client, LAPTOP)
    response = client.post(f"/api/cases/{LAPTOP}/questions", json={"question": "Who came back?"})
    assert response.status_code == 409 and "rate limit" in response.json()["detail"]


# --- no credentials at all --------------------------------------------------------


def keyless_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARIABLES}
    return env | {"PYTHONPATH": str(ROOT), "HOME": str(home), "USERPROFILE": str(home), "PYTHONIOENCODING": "utf-8"}


def cli(tmp_path, *args):
    command = [sys.executable, "-m", "detective", *args]
    return subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, env=keyless_env(tmp_path))


@pytest.mark.parametrize(
    "arguments",
    [("solve", LAPTOP), ("ask", LAPTOP, "Who?"), ("uncover", LAPTOP, "L1"), ("evaluate",)],
)
def test_cli_explains_a_missing_key_without_a_traceback(tmp_path, arguments):
    result = cli(tmp_path, *arguments, "--backend", "claude")
    assert result.returncode != 0
    assert "ANTHROPIC_API_KEY" in result.stderr and "Traceback" not in result.stderr


def test_offline_commands_work_with_no_credentials(tmp_path):
    result = cli(tmp_path, "solve", CODEX)
    assert result.returncode == 0, result.stderr
    assert "Helena Marsh" in result.stdout and (tmp_path / "runs" / CODEX / "report.md").exists()


def test_web_app_works_offline_and_refuses_claude_with_no_credentials(tmp_path, monkeypatch):
    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert claude_configured() is False

    client = TestClient(create_app(data_dir=tmp_path / "data"), headers=HEADERS)
    assert client.get("/api/health").json()["claude_configured"] is False
    refused = client.post(f"/api/cases/{LAPTOP}/runs", json={"backend": "claude"})
    assert refused.status_code == 409 and "ANTHROPIC_API_KEY" in refused.json()["detail"]
    assert client.post(f"/api/cases/{LAPTOP}/runs", json={"backend": "offline"}).status_code == 202


@pytest.mark.parametrize("variable", KEY_VARIABLES)
def test_each_credential_source_counts_as_configured(variable, tmp_path, monkeypatch):
    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv(variable, "x")
    assert claude_configured() is True


def test_saved_login_profile_counts_as_configured(tmp_path, monkeypatch):
    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / ".config" / "anthropic").mkdir(parents=True)
    assert claude_configured() is True


def test_live_tests_skip_rather_than_fail_with_no_key():
    source = (ROOT / "tests" / "test_live.py").read_text(encoding="utf-8")
    assert "skipif" in source and "ANTHROPIC_API_KEY" in source
