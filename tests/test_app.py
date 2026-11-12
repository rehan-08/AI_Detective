"""The application end to end through its HTTP API, without a browser."""
from __future__ import annotations

import io
import json
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import CASES
from detective.claude_backend import ClaudeExtractor, ClaudeGateway, ClaudeReasoner
from detective.heuristic import HeuristicExtractor
from detective.models import Citation, DocExtraction
from detective.rules import RuleReasoner
from detective.server import create_app
from detective.workspace import Backends

HEADERS = {"x-detective-client": "web"}
LAPTOP, NECKLACE, CODEX = "case_01_missing_laptop", "case_02_aldermere_necklace", "case_03_halden_codex"


def client_for(data_dir: Path, backends: Backends | None = None) -> TestClient:
    return TestClient(create_app(data_dir=data_dir, backends=backends), headers=HEADERS)


@pytest.fixture
def client(tmp_path):
    return client_for(tmp_path)


def run_case(client: TestClient, case_id: str, auto_leads: bool = True, backend: str = "offline") -> dict:
    response = client.post(f"/api/cases/{case_id}/runs", json={"backend": backend, "auto_leads": auto_leads})
    assert response.status_code == 202, response.text
    client.app.state.manager.get(case_id).wait()
    return client.get(f"/api/cases/{case_id}/state").json()


DRAFT = {
    "title": "The Cookie Jar",
    "briefing": "Someone emptied the cookie jar.",
    "crime_summary": "Cookies were taken from the Kitchen.",
    "scene": "Kitchen",
    "window_start": "2025-01-10T15:00",
    "window_end": "2025-01-10T16:00",
    "suspects": [
        {"name": "Ana Bell", "role": "Sister", "aliases": ["Ana"]},
        {"name": "Ben Cole", "role": "Brother", "aliases": ["Ben"]},
    ],
    "locations": [{"name": "House"}, {"name": "Kitchen", "within": "House"}, {"name": "Garden", "within": "House"}],
    "documents": [
        {
            "title": "Door sensor",
            "type": "system_log",
            "date": "2025-01-10",
            "device": "Kitchen sensor",
            "text": "15:10 | Ana Bell | Kitchen | ENTRY\n15:20 | Ana Bell | Kitchen | EXIT\n14:30 | Ben Cole | Garden | ENTRY\n16:30 | Ben Cole | Garden | EXIT",
        },
        {
            "title": "Interview with Ana",
            "type": "witness_interview",
            "date": "2025-01-11",
            "speaker": "Ana Bell",
            "text": "A: I was in the Garden from 15:00 until 16:00.",
        },
    ],
}


def case_zip(*, tweak=None, extra: dict[str, bytes] | None = None, folder: str = "my_case/") -> bytes:
    source = CASES / LAPTOP
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        manifest = json.loads((source / "case.json").read_text(encoding="utf-8"))
        if tweak:
            tweak(manifest)
        archive.writestr(f"{folder}case.json", json.dumps(manifest))
        for path in [*source.glob("documents/*.txt"), *source.glob("sealed/*.txt")]:
            archive.writestr(f"{folder}{path.relative_to(source).as_posix()}", path.read_bytes())
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return buffer.getvalue()


def post_zip(client, data):
    return client.post("/api/cases/import", content=data, headers={"content-type": "application/zip"})


# --- the investigator's journey ------------------------------------------------


def test_serves_the_application_and_health(client):
    page = client.get("/")
    assert page.status_code == 200 and "AI Detective" in page.text
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["model"] == "claude-opus-5-5"


def test_lists_the_built_in_cases(client):
    cases = client.get("/api/cases").json()
    assert {c["id"] for c in cases} == {LAPTOP, NECKLACE, CODEX}
    assert all(c["status"] == "idle" and c["culprit"] is None for c in cases)


def test_full_case_journey(client):
    state = run_case(client, CODEX, auto_leads=False)
    assert state["status"] == "done"
    assert state["conclusion"]["culprit"] == "Elias Brandt"
    assert {lead["id"] for lead in state["pending_leads"]} == {"L1", "L2", "L3"}
    assert state["suspects"] and state["timeline"] and state["contradictions"] and state["findings"]

    for lead in ("L1", "L2", "L3"):
        assert client.post(f"/api/cases/{CODEX}/leads/{lead}/uncover").status_code == 202
        client.app.state.manager.get(CODEX).wait()
    state = client.get(f"/api/cases/{CODEX}/state").json()
    assert state["conclusion"]["culprit"] == "Helena Marsh"
    assert state["conclusion"]["confidence"] == "HIGH"
    assert [r["leader"] for r in state["rounds"]][0] == "Elias Brandt"
    assert state["opened_leads"] == ["L1", "L2", "L3"]

    answer = client.post(f"/api/cases/{CODEX}/questions", json={"question": "Who is on the Brandt landline call to Greta Brandt?"}).json()
    assert any(c["passage_id"].startswith("ES1") for c in answer["citations"])

    report = client.get(f"/api/cases/{CODEX}/report.md")
    assert report.headers["content-disposition"].startswith("attachment")
    assert "Helena Marsh" in report.text

    evaluation = client.get(f"/api/cases/{CODEX}/evaluation").json()
    assert evaluation["passed"] is True


def test_every_built_in_case_passes_through_the_api(client):
    for case_id in (LAPTOP, NECKLACE, CODEX):
        run_case(client, case_id)
        assert client.get(f"/api/cases/{case_id}/evaluation").json()["passed"]


def test_session_survives_a_restart_without_repeating_work(tmp_path):
    first = client_for(tmp_path)
    run_case(first, CODEX)
    first.post(f"/api/cases/{CODEX}/questions", json={"question": "Who is Helena Marsh?"})
    before = first.get(f"/api/cases/{CODEX}/state").json()

    second = client_for(tmp_path)
    after = second.get(f"/api/cases/{CODEX}/state").json()
    for key in ("status", "conclusion", "findings", "rounds", "opened_leads", "questions", "timeline", "contradictions"):
        assert after[key] == before[key], key
    summary = next(c for c in second.get("/api/cases").json() if c["id"] == CODEX)
    assert summary["culprit"] == "Helena Marsh"
    assert second.get(f"/api/cases/{CODEX}/documents/ES1").status_code == 200


def test_reset_clears_the_session_and_reseals_leads(client):
    run_case(client, LAPTOP)
    assert client.get(f"/api/cases/{LAPTOP}/documents/S1").status_code == 200
    assert client.post(f"/api/cases/{LAPTOP}/reset").status_code == 204
    assert client.get(f"/api/cases/{LAPTOP}/state").json()["conclusion"] is None
    assert client.get(f"/api/cases/{LAPTOP}/documents/S1").status_code == 404
    assert len(client.get(f"/api/cases/{LAPTOP}/state").json()["pending_leads"]) == 1


def test_evidence_browsing_and_search_before_any_run(client):
    documents = client.get(f"/api/cases/{LAPTOP}/documents").json()
    assert "D2" in {d["id"] for d in documents} and "S1" not in {d["id"] for d in documents}
    assert client.get(f"/api/cases/{LAPTOP}/documents/D2").json()["passages"]
    assert client.get(f"/api/cases/{LAPTOP}/search", params={"q": "laptop sleeve"}).json()[0]["doc_id"] == "D8"
    assert not client.get(f"/api/cases/{LAPTOP}/search", params={"q": "QuickCash"}).json()


# --- refusals ------------------------------------------------------------------


def test_actions_out_of_order_are_refused_clearly(client):
    assert client.post(f"/api/cases/{LAPTOP}/leads/L1/uncover").status_code == 409
    assert client.post(f"/api/cases/{LAPTOP}/questions", json={"question": "Who?"}).status_code == 409
    assert client.get(f"/api/cases/{LAPTOP}/report.md").status_code == 409
    assert client.get(f"/api/cases/{LAPTOP}/evaluation").status_code == 409
    assert client.get("/api/cases/nope").status_code == 404


def test_unknown_or_used_lead_is_refused(client):
    run_case(client, LAPTOP)
    assert client.post(f"/api/cases/{LAPTOP}/leads/L1/uncover").status_code == 404
    assert client.post(f"/api/cases/{LAPTOP}/leads/L9/uncover").status_code == 404


def test_claude_is_refused_when_not_configured(tmp_path):
    backends = Backends(claude_available=lambda: False)
    client = client_for(tmp_path, backends)
    response = client.post(f"/api/cases/{LAPTOP}/runs", json={"backend": "claude"})
    assert response.status_code == 409 and "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert client.get("/api/health").json()["claude_configured"] is False


def test_unknown_backend_is_refused(client):
    assert client.post(f"/api/cases/{LAPTOP}/runs", json={"backend": "gpt"}).status_code == 404


def test_second_run_is_refused_while_one_is_in_progress(tmp_path):
    release = threading.Event()

    class SlowExtractor:
        name = "slow"

        def extract(self, doc):
            release.wait(10)
            return DocExtraction(doc_id=doc.id, entities=[], events=[], signals=[], defeaters=[], credentials=[], identifiers=[])

    class Reasoner:
        def investigate(self, inv, auto_leads=True):
            inv.reassess("start")

    backends = Backends(claude_factory=lambda case: (SlowExtractor(), Reasoner()), claude_available=lambda: True)
    client = client_for(tmp_path, backends)
    assert client.post(f"/api/cases/{LAPTOP}/runs", json={"backend": "claude"}).status_code == 202
    assert client.post(f"/api/cases/{LAPTOP}/runs", json={"backend": "offline"}).status_code == 409
    assert client.post(f"/api/cases/{LAPTOP}/reset").status_code == 409
    assert client.get(f"/api/cases/{LAPTOP}/state").json()["status"] == "running"
    release.set()
    client.app.state.manager.get(LAPTOP).wait()
    assert client.get(f"/api/cases/{LAPTOP}/state").json()["status"] == "done"


def test_a_failed_run_ends_in_a_visible_error_and_can_be_retried(tmp_path):
    class Broken:
        name = "broken"

        def extract(self, doc):
            raise RuntimeError("the extractor broke")

    backends = Backends(claude_factory=lambda case: (Broken(), None), claude_available=lambda: True)
    client = client_for(tmp_path, backends)
    state = run_case(client, LAPTOP, backend="claude")
    assert state["status"] == "failed" and "broke" in state["error"]
    assert run_case(client, LAPTOP)["status"] == "done"


def test_corrupt_session_does_not_stop_the_app(tmp_path):
    first = client_for(tmp_path)
    run_case(first, LAPTOP)
    (tmp_path / "sessions" / f"{LAPTOP}.json").write_text("{not json", encoding="utf-8")
    second = client_for(tmp_path)
    assert second.get(f"/api/cases/{LAPTOP}/state").json()["conclusion"] is None
    assert run_case(second, LAPTOP)["status"] == "done"


def test_broken_custom_case_does_not_stop_the_app(tmp_path):
    folder = tmp_path / "cases" / "broken"
    folder.mkdir(parents=True)
    (folder / "case.json").write_text("{not json", encoding="utf-8")
    assert len(client_for(tmp_path).get("/api/cases").json()) == 3


def test_failed_question_gives_a_reason(tmp_path):
    class BrokenReasoner:
        def investigate(self, inv, auto_leads=True):
            RuleReasoner().investigate(inv, auto_leads=auto_leads)

        def answer(self, inv, question):
            raise RuntimeError("the model is unavailable")

    backends = Backends(claude_factory=lambda case: (HeuristicExtractor(case), BrokenReasoner()), claude_available=lambda: True)
    client = client_for(tmp_path, backends)
    run_case(client, LAPTOP, backend="claude")
    response = client.post(f"/api/cases/{LAPTOP}/questions", json={"question": "Who?"})
    assert response.status_code == 409 and "unavailable" in response.json()["detail"]


def test_front_end_never_writes_raw_html():
    script = (Path(__file__).resolve().parent.parent / "detective" / "web" / "app.js").read_text(encoding="utf-8")
    assert "innerHTML =" not in script and "insertAdjacentHTML" not in script and "document.write" not in script


def test_claude_path_runs_through_the_app(tmp_path):
    """The real Claude classes, with a scripted transport in place of the API."""

    class Block:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    class Usage:
        input_tokens = output_tokens = 1

    def tool(name, arguments):
        return Block(type="tool_use", id=f"toolu_{name}", name=name, input=arguments)

    finding = {
        "statement": "Sam returned at 19:42.",
        "status": "fact",
        "citations": [{"passage_id": "D8:p2", "quote": "Sam Whitlock enters"}],
        "rationale": "",
    }
    suspects = ["Tom Reyes", "Lena Fischer", "Priya Nair"]
    conclusion = {
        "culprit": "Sam Whitlock",
        "accomplices": [],
        "confidence": "HIGH",
        "confidence_explanation": "Records agree.",
        "narrative": "He came back.",
        "supporting_finding_ids": ["F1"],
        "eliminated": [{"name": n, "reason": "Excluded."} for n in suspects],
        "disagreement_with_engine": "The scripted extraction is empty.",
    }
    turns = [[tool("record_finding", finding)], [tool("submit_conclusion", conclusion)]]

    class Messages:
        def create(self, **_):
            return Block(content=turns.pop(0), stop_reason="tool_use", usage=Usage)

        def parse(self, **request):
            doc = request["messages"][0]["content"].split("doc_id: ")[1].split("\n")[0]
            empty = DocExtraction(doc_id=doc, entities=[], events=[], signals=[], defeaters=[], credentials=[], identifiers=[])
            return Block(parsed_output=empty, stop_reason="end_turn", usage=Usage)

    fake = Block(beta=Block(messages=Messages()))

    def factory(case):
        gateway = ClaudeGateway(client=fake)
        return ClaudeExtractor(case, gateway, cache_dir=None), ClaudeReasoner(gateway)

    client = client_for(tmp_path, Backends(claude_factory=factory, claude_available=lambda: True))
    state = run_case(client, LAPTOP, backend="claude")
    assert state["status"] == "done", state["error"]
    assert state["backend"] == "claude" and state["conclusion"]["culprit"] == "Sam Whitlock"
    assert state["conclusion"]["engine_agrees"] is False


def test_claude_rerun_starts_a_fresh_conclusion(tmp_path):
    """A follow-up run must not exit early on the previous run's conclusion."""
    client = client_for(tmp_path)
    run_case(client, LAPTOP, auto_leads=False)
    inv = client.app.state.manager.get(LAPTOP).inv
    calls = []

    class Gateway:
        def turn(self, *_):
            calls.append(1)
            raise RuntimeError("reached the model")

    with pytest.raises(RuntimeError, match="reached the model"):
        ClaudeReasoner(Gateway()).investigate(inv, auto_leads=False)
    assert calls and inv.conclusion is None


# --- creating and importing cases ----------------------------------------------


def test_create_a_case_and_solve_it(client):
    created = client.post("/api/cases", json=DRAFT)
    assert created.status_code == 201, created.text
    case_id = created.json()["id"]
    assert case_id == "the_cookie_jar"
    state = run_case(client, case_id)
    assert state["conclusion"]["culprit"] == "Ana Bell"
    assert any(c["person"] == "Ana Bell" for c in state["contradictions"])
    assert client.get(f"/api/cases/{case_id}/evaluation").status_code == 404, "no key was supplied"


def test_custom_case_survives_a_restart_and_can_be_deleted(tmp_path):
    first = client_for(tmp_path)
    case_id = first.post("/api/cases", json=DRAFT).json()["id"]
    run_case(first, case_id)
    second = client_for(tmp_path)
    assert second.get(f"/api/cases/{case_id}").status_code == 200
    assert second.get(f"/api/cases/{case_id}/state").json()["conclusion"]["culprit"] == "Ana Bell"
    assert second.delete(f"/api/cases/{case_id}").status_code == 204
    assert second.get(f"/api/cases/{case_id}").status_code == 404
    assert client_for(tmp_path).get(f"/api/cases/{case_id}").status_code == 404


def test_built_in_cases_cannot_be_deleted(client):
    assert client.delete(f"/api/cases/{LAPTOP}").status_code == 409
    assert (CASES / LAPTOP / "case.json").exists()


def test_same_title_twice_gets_a_new_id(client):
    first = client.post("/api/cases", json=DRAFT).json()["id"]
    second = client.post("/api/cases", json=DRAFT).json()["id"]
    assert first != second


@pytest.mark.parametrize(
    "change, message",
    [
        ({"scene": "Attic"}, "must be one of"),
        ({"window_start": "yesterday"}, "must look like"),
        ({"window_end": "2025-01-10T14:00"}, "must end after"),
        ({"suspects": []}, "at least one suspect"),
        ({"documents": []}, "at least one document"),
    ],
)
def test_invalid_cases_are_refused_with_a_reason(client, change, message):
    response = client.post("/api/cases", json=DRAFT | change)
    assert response.status_code == 422 and message in response.json()["detail"]


def test_unknown_document_type_is_refused(client):
    document = DRAFT["documents"][0] | {"type": "tea_leaves"}
    assert client.post("/api/cases", json=DRAFT | {"documents": [document]}).status_code == 422


def test_import_a_zip_and_solve_it(client):
    response = post_zip(client, case_zip())
    assert response.status_code == 201, response.text
    case_id = response.json()["id"]
    assert case_id != LAPTOP
    assert run_case(client, case_id)["conclusion"]["culprit"] == "Sam Whitlock"
    assert client.get(f"/api/cases/{case_id}/documents/S1").status_code == 200


def test_import_refuses_bad_archives(client, tmp_path):
    assert post_zip(client, b"not a zip").status_code == 422

    missing_manifest = io.BytesIO()
    with zipfile.ZipFile(missing_manifest, "w") as archive:
        archive.writestr("notes.txt", "hello")
    assert "case.json" in post_zip(client, missing_manifest.getvalue()).json()["detail"]

    assert post_zip(client, case_zip(extra={"my_case/run.exe": b"x"})).status_code == 422

    def bad_scene(manifest):
        manifest["crime"]["location"] = "Nowhere"

    assert "must be one of" in post_zip(client, case_zip(tweak=bad_scene)).json()["detail"]


@pytest.mark.parametrize("name", ["../evil.txt", "..\\evil.txt", "my_case/../../evil.txt", "C:/evil.txt", "/evil.txt"])
def test_import_refuses_path_traversal(client, name):
    response = post_zip(client, case_zip(extra={name: b"x"}))
    assert response.status_code == 422 and "Unsafe path" in response.json()["detail"]


def test_import_refuses_a_lead_pointing_outside_the_case(client):
    def escape(manifest):
        manifest["leads"][0]["document"] = "../../secret.txt"

    response = post_zip(client, case_zip(tweak=escape))
    assert response.status_code == 422 and "Unsafe lead path" in response.json()["detail"]


def test_import_refuses_an_oversized_archive(client):
    assert post_zip(client, b"0" * 10_000_001).status_code == 422


def test_failed_imports_leave_nothing_behind(client, tmp_path):
    post_zip(client, case_zip(extra={"../evil.txt": b"x"}))
    custom = tmp_path / "cases"
    assert not custom.exists() or not list(custom.iterdir())


# --- local security ------------------------------------------------------------


def test_writes_need_the_client_header(tmp_path):
    plain = TestClient(create_app(data_dir=tmp_path))
    assert plain.post(f"/api/cases/{LAPTOP}/runs", json={}).status_code == 403
    assert plain.post("/api/cases/import", content=b"x").status_code == 403
    assert plain.delete(f"/api/cases/{LAPTOP}").status_code == 403
    assert plain.get("/api/cases").status_code == 200


def test_no_cors_is_offered_to_other_sites(client):
    response = client.options(
        f"/api/cases/{LAPTOP}/runs",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_rebound_host_is_refused(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path), base_url="http://evil.example")
    assert client.get("/api/cases").status_code == 400


def test_answer_key_is_never_exposed_before_evaluation(client):
    run_case(client, CODEX, auto_leads=False)
    overview = client.get(f"/api/cases/{CODEX}").text + client.get(f"/api/cases/{CODEX}/state").text
    assert "red_herring" not in overview and "solution.json" not in overview


def test_failure_message_for_a_bad_key():
    import anthropic
    import httpx

    from detective.workspace import describe_failure

    response = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com"))
    error = anthropic.AuthenticationError("nope", response=response, body=None)
    assert "API key" in describe_failure(error)


def test_citation_round_trip_in_state(client):
    state = run_case(client, LAPTOP)
    Citation.model_validate(state["findings"][0]["citations"][0])
