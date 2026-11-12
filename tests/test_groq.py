"""The Groq path, through the real Groq SDK against a local fake of its API.

As with the Claude tests, this proves the translation layer and guards, not the
quality of Groq's models. tests/test_live.py measures that when a key is set.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import groq
import pytest
from fastapi.testclient import TestClient

from conftest import CASE_NAMES, CASES
from fake_claude import DetectiveModel, FakeApi, api_error, chat_reply, message, on_groq, scripted, text, tool_use
from detective.evaluation import load_solution, score
from detective.groq_backend import DEFAULT_MODEL, GroqExtractor, GroqGateway, GroqReasoner, groq_model, strict_schema, to_chat_messages
from detective.ingest import load_case
from detective.investigation import Investigation
from detective.models import DocExtraction
from detective.server import create_app
from detective.workspace import Backends, describe_failure

ROOT = Path(__file__).resolve().parent.parent
LAPTOP, CODEX = "case_01_missing_laptop", "case_03_halden_codex"
HEADERS = {"x-detective-client": "web"}


@pytest.fixture
def fake_groq():
    created = []

    def make(respond):
        api = FakeApi(respond)
        created.append(api)
        return api

    yield make
    for api in created:
        api.close()


def gateway_for(api) -> GroqGateway:
    return GroqGateway(client=groq.Groq(api_key="fake-key-for-tests", base_url=api.url, max_retries=0))


def investigation(case, api):
    gateway = gateway_for(api)
    return Investigation(case, GroqExtractor(case, gateway, cache_dir=None)), GroqReasoner(gateway)


def app_client(tmp_path, api) -> TestClient:
    def factory(case):
        gateway = gateway_for(api)
        return GroqExtractor(case, gateway, cache_dir=None), GroqReasoner(gateway)

    backends = Backends(groq_factory=factory, groq_available=lambda: True)
    return TestClient(create_app(data_dir=tmp_path, backends=backends), headers=HEADERS)


def run(client, case_id, auto_leads=True) -> dict:
    response = client.post(f"/api/cases/{case_id}/runs", json={"backend": "groq", "auto_leads": auto_leads})
    assert response.status_code == 202, response.text
    client.app.state.manager.get(case_id).wait()
    return client.get(f"/api/cases/{case_id}/state").json()


EMPTY = DocExtraction(doc_id="x", entities=[], events=[], signals=[], defeaters=[], credentials=[], identifiers=[]).model_dump_json()


def agent_script(replies):
    """Answer extraction with nothing, then play scripted agent turns."""
    queue = list(replies)

    def respond(request):
        if "response_format" in request:
            return chat_reply(text(EMPTY))
        return chat_reply(queue.pop(0)) if isinstance(queue[0], dict) and "content" in queue[0] else queue.pop(0)

    return respond


def good_finding():
    return {
        "statement": "Sam came back.",
        "status": "fact",
        "citations": [{"passage_id": "D8:p2", "quote": "Sam Whitlock enters"}],
        "rationale": "CCTV.",
    }


def conclusion():
    names = [p.name for p in load_case(CASES / LAPTOP).suspects if p.name != "Sam Whitlock"]
    return {
        "culprit": "Sam Whitlock",
        "accomplices": [],
        "confidence": "MEDIUM",
        "confidence_explanation": "Stated.",
        "narrative": "Stated.",
        "supporting_finding_ids": ["F1"],
        "eliminated": [{"name": n, "reason": "Lower."} for n in names],
        "disagreement_with_engine": "The scripted extraction is empty.",
    }


# --- whole cases ------------------------------------------------------------------


@pytest.mark.parametrize("name", CASE_NAMES)
def test_groq_path_completes_each_case_with_no_key(name, fake_groq):
    case = load_case(CASES / name)
    inv, reasoner = investigation(case, fake_groq(on_groq(DetectiveModel(case))))
    reasoner.investigate(inv)
    result = score(inv, load_solution(CASES / name))
    assert result.culprit_correct, result.misses
    assert result.citation_validity == 1.0 and result.red_herrings_clean
    assert result.contradiction_recall == 1.0 and result.timeline_recall == 1.0
    assert not inv.rejections and not inv.finding_rejections


def test_app_journey_on_the_groq_path(fake_groq, tmp_path):
    case = load_case(CASES / CODEX)
    client = app_client(tmp_path, fake_groq(on_groq(DetectiveModel(case))))
    state = run(client, CODEX, auto_leads=False)
    assert state["status"] == "done", state["error"]
    assert state["backend"] == "groq" and state["conclusion"]["culprit"] == "Elias Brandt"

    assert client.post(f"/api/cases/{CODEX}/leads/L1/uncover").status_code == 202
    client.app.state.manager.get(CODEX).wait()
    state = client.get(f"/api/cases/{CODEX}/state").json()
    assert state["status"] == "done", state["error"]
    assert state["conclusion"]["culprit"] == "Helena Marsh"


def test_saved_groq_session_reopens_without_model_calls(fake_groq, tmp_path):
    case = load_case(CASES / LAPTOP)
    api = fake_groq(on_groq(DetectiveModel(case)))
    run(app_client(tmp_path, api), LAPTOP)
    calls = len(api.requests)
    state = app_client(tmp_path, api).get(f"/api/cases/{LAPTOP}/state").json()
    assert state["backend"] == "groq" and state["conclusion"]["culprit"] == "Sam Whitlock"
    assert len(api.requests) == calls


# --- request shape ----------------------------------------------------------------


def test_requests_match_groqs_documented_shape(fake_groq):
    case = load_case(CASES / LAPTOP)
    api = fake_groq(on_groq(DetectiveModel(case)))
    inv, reasoner = investigation(case, api)
    reasoner.investigate(inv)

    extraction = next(r for r in api.requests if "response_format" in r)
    assert extraction["model"] == DEFAULT_MODEL
    assert extraction["response_format"]["type"] == "json_schema"
    assert extraction["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in extraction, "Groq does not allow tools alongside structured outputs"

    turns = [r for r in api.requests if "tools" in r]
    assert turns[0]["messages"][0]["role"] == "system"
    tool = turns[0]["tools"][0]
    assert tool["type"] == "function" and {"name", "description", "parameters"} == set(tool["function"])
    results = [m for r in turns for m in r["messages"] if m["role"] == "tool"]
    assert results and all(m["tool_call_id"] and m["name"] and isinstance(m["content"], str) for m in results)
    sent = json.dumps(api.requests)
    for anthropic_only in ("fallbacks", "cache_control", "input_schema", "tool_use_id"):
        assert anthropic_only not in sent


def test_strict_schema_stands_alone():
    schema = strict_schema(DocExtraction)
    text_form = json.dumps(schema)
    assert "$ref" not in text_form and "$defs" not in text_form
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    event = schema["properties"]["events"]["items"]
    assert event["additionalProperties"] is False and set(event["required"]) == set(event["properties"])


def test_every_call_in_a_turn_gets_its_own_tool_message():
    blocks = [
        type("B", (), {"type": "text", "text": "Checking."})(),
        type("B", (), {"type": "tool_use", "id": "a", "name": "first", "input": {"x": 1}})(),
        type("B", (), {"type": "tool_use", "id": "b", "name": "second", "input": {}})(),
    ]
    results = [
        {"type": "tool_result", "tool_use_id": "a", "content": "one"},
        {"type": "tool_result", "tool_use_id": "b", "content": "bad", "is_error": True},
    ]
    chat = to_chat_messages("sys", [{"role": "user", "content": "go"}, {"role": "assistant", "content": blocks}, {"role": "user", "content": results}])
    assert [m["role"] for m in chat] == ["system", "user", "assistant", "tool", "tool"]
    assert chat[2]["content"] == "Checking." and json.loads(chat[2]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert (chat[3]["name"], chat[4]["name"]) == ("first", "second")


def test_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/new-model")
    assert groq_model() == "some/new-model"
    monkeypatch.delenv("GROQ_MODEL")
    assert groq_model() == DEFAULT_MODEL


# --- the same guards apply --------------------------------------------------------


def test_fabricated_extraction_is_rejected(fake_groq):
    case = load_case(CASES / LAPTOP)
    invented = json.loads(EMPTY) | {"signals": [{
        "person": "Sam Whitlock", "kind": "motive", "detail": "Greed",
        "citation": {"passage_id": "D1:p1", "quote": "Sam is greedy"},
    }]}
    inv, _ = investigation(case, fake_groq(lambda _r: chat_reply(text(json.dumps(invented)))))
    assert len(inv.rejections) == len(case.documents)
    assert not any(e.signals for e in inv.extractions.values())


def test_fabricated_finding_is_returned_to_the_model(fake_groq):
    case = load_case(CASES / LAPTOP)
    fake = good_finding() | {"citations": [{"passage_id": "D5:p1", "quote": "I confess"}]}
    api = fake_groq(agent_script([
        message([tool_use(1, "record_finding", fake)], "tool_use"),
        message([tool_use(2, "record_finding", good_finding())], "tool_use"),
        message([tool_use(3, "submit_conclusion", conclusion())], "tool_use"),
    ]))
    inv, reasoner = investigation(case, api)
    assert reasoner.investigate(inv).culprit == "Sam Whitlock"
    assert len(inv.findings) == 1 and len(inv.finding_rejections) == 1
    assert any(m["role"] == "tool" and "rejected" in m["content"] for r in api.requests for m in r["messages"])


def test_unparseable_tool_arguments_go_back_as_an_error(fake_groq):
    case = load_case(CASES / LAPTOP)
    broken = chat_reply(message([tool_use(1, "search_evidence", {})], "tool_use"))
    broken["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    api = fake_groq(agent_script([
        broken,
        message([tool_use(2, "record_finding", good_finding())], "tool_use"),
        message([tool_use(3, "submit_conclusion", conclusion())], "tool_use"),
    ]))
    inv, reasoner = investigation(case, api)
    reasoner.investigate(inv)
    assert any(m["role"] == "tool" and "invalid arguments" in m["content"] for r in api.requests for m in r["messages"])


def test_rejected_tool_call_is_retried(fake_groq):
    case = load_case(CASES / LAPTOP)
    failed = (400, {"error": {"message": "Failed to call a function", "type": "invalid_request_error", "code": "tool_use_failed"}})
    api = fake_groq(agent_script([
        failed,
        message([tool_use(1, "record_finding", good_finding())], "tool_use"),
        message([tool_use(2, "submit_conclusion", conclusion())], "tool_use"),
    ]))
    inv, reasoner = investigation(case, api)
    assert reasoner.investigate(inv).culprit == "Sam Whitlock"


def test_other_bad_requests_are_not_retried(fake_groq):
    case = load_case(CASES / LAPTOP)
    api = fake_groq(agent_script([(400, {"error": {"message": "model not found", "type": "invalid_request_error"}})]))
    inv, reasoner = investigation(case, api)
    with pytest.raises(groq.BadRequestError):
        reasoner.investigate(inv)


# --- failures ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [
        (api_error(401, "authentication_error"), "rejected the API key"),
        (api_error(429, "rate_limit_error"), "rate limit"),
        (api_error(500), "error (500)"),
        (chat_reply(message([], "max_tokens")), "truncated"),
        (chat_reply(message([], "content_filter")), "declined"),
        (chat_reply(text("this is not json")), "does not match the schema"),
    ],
)
def test_failures_during_a_run_are_explained(fake_groq, tmp_path, reply, expected):
    state = run(app_client(tmp_path, fake_groq(lambda _r: reply)), LAPTOP)
    assert state["status"] == "failed" and expected in state["error"]


def test_unreachable_service_is_explained(fake_groq, tmp_path):
    api = fake_groq(lambda _r: chat_reply(text("unused")))
    client = app_client(tmp_path, api)
    api.close()
    state = run(client, LAPTOP)
    assert state["status"] == "failed" and "reach the Groq API" in state["error"]


# --- configuration ----------------------------------------------------------------


def test_refused_when_no_groq_key_is_set(tmp_path):
    backends = Backends(groq_available=lambda: False)
    client = TestClient(create_app(data_dir=tmp_path, backends=backends), headers=HEADERS)
    health = client.get("/api/health").json()
    assert health["groq_configured"] is False and health["groq_model"]
    response = client.post(f"/api/cases/{LAPTOP}/runs", json={"backend": "groq"})
    assert response.status_code == 409 and "GROQ_API_KEY" in response.json()["detail"]


def test_cli_explains_a_missing_groq_key(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"} | {"PYTHONPATH": str(ROOT)}
    command = [sys.executable, "-m", "detective", "solve", LAPTOP, "--backend", "groq"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "GROQ_API_KEY" in result.stderr and "Traceback" not in result.stderr


def test_providers_are_independent(tmp_path):
    backends = Backends(claude_available=lambda: False, groq_available=lambda: True)
    assert backends.groq_available() and not backends.claude_available()


def test_groq_failures_do_not_mention_claude():
    import httpx

    response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com"))
    assert "Groq" in describe_failure(groq.AuthenticationError("nope", response=response, body=None))


def test_no_key_is_stored_in_the_project():
    tracked = [*ROOT.glob("detective/**/*.py"), *ROOT.glob("detective/web/*"), ROOT / "README.md", ROOT / "pyproject.toml"]
    assert not any("gsk_" in path.read_text(encoding="utf-8") for path in tracked)
