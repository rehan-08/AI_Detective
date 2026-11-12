"""Exercise the Claude path through the real SDK against a local stub of the API.

The stub checks request shape and lets us script model behaviour, including a model
that fabricates evidence. It cannot show how well Claude solves a case; test_live.py
does that when credentials are available.
"""
from __future__ import annotations

import json

import pytest

from conftest import CASES
from fake_claude import FakeApi, message, scripted, tool_use
from detective.claude_backend import ClaudeExtractor, ClaudeReasoner, RefusedError
from detective.heuristic import HeuristicExtractor
from detective.ingest import load_case
from detective.investigation import Investigation
from detective.models import Confidence, DocExtraction, Status
from detective.rules import RuleReasoner
from detective.tools import InvestigatorTools, ToolError


@pytest.fixture
def api():
    stubs = []

    def make(script):
        stub = FakeApi(scripted(script))
        stubs.append(stub)
        return stub

    yield make
    for stub in stubs:
        stub.close()


class EmptyExtractor:
    name = "empty"

    def extract(self, doc):
        return DocExtraction(doc_id=doc.id, entities=[], events=[], signals=[], defeaters=[], credentials=[], identifiers=[])


def case_one():
    return load_case(CASES / "case_01_missing_laptop")


EXTRACTION = {
    "doc_id": "D5",
    "entities": [],
    "events": [
        {
            "description": "Sam says he went home",
            "person": "Sam Whitlock",
            "credential": None,
            "location": "Home of Sam Whitlock",
            "start": "2025-03-13T18:05",
            "end": "2025-03-13T23:59",
            "present": True,
            "is_presence": True,
            "approximate": False,
            "citation": {"passage_id": "D5:p1", "quote": "I left the studio at 18:05"},
        },
        {
            "description": "Invented confession",
            "person": "Sam Whitlock",
            "credential": None,
            "location": "Studio B",
            "start": "2025-03-13T19:45",
            "end": None,
            "present": True,
            "is_presence": True,
            "approximate": False,
            "citation": {"passage_id": "D5:p1", "quote": "I took the laptop at 19:45"},
        },
    ],
    "signals": [],
    "defeaters": [],
    "credentials": [],
    "identifiers": [],
}


def test_extraction_request_shape_and_fabrication_filter(api):
    stub = api([message([{"type": "text", "text": json.dumps(EXTRACTION)}])])
    case = case_one()
    case.documents = [d for d in case.documents if d.id == "D5"]
    inv = Investigation(case, ClaudeExtractor(case, stub.gateway(), cache_dir=None))

    assert len(inv.extractions["D5"].events) == 1, "the fabricated event must not be admitted"
    assert len(inv.rejections) == 1 and "not found" in inv.rejections[0].reason

    request = stub.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert request["output_config"]["effort"] == "high"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "server-side-fallback-2026-07-01" in stub.headers[0]["anthropic-beta"]
    assert "temperature" not in request and "budget_tokens" not in json.dumps(request)


def test_extraction_is_cached_between_runs(api, tmp_path):
    stub = api([message([{"type": "text", "text": json.dumps(EXTRACTION)}])])
    case = case_one()
    doc = next(d for d in case.documents if d.id == "D5")
    extractor = ClaudeExtractor(case, stub.gateway(), cache_dir=tmp_path)
    first = extractor.extract(doc)
    second = extractor.extract(doc)
    assert first == second and len(stub.requests) == 1


def test_refusal_is_surfaced(api):
    reply = message([], stop_reason="refusal") | {"stop_details": {"type": "refusal", "category": None, "explanation": "no"}}
    stub = api([reply])
    case = case_one()
    with pytest.raises(RefusedError):
        ClaudeExtractor(case, stub.gateway(), cache_dir=None).extract(case.documents[0])


def conclusion(culprit="Sam Whitlock", findings=("F1",), disagreement=""):
    others = [p.name for p in case_one().suspects if p.name != culprit]
    return {
        "culprit": culprit,
        "accomplices": [],
        "confidence": "MEDIUM",
        "confidence_explanation": "Two records.",
        "narrative": "He came back.",
        "supporting_finding_ids": list(findings),
        "eliminated": [{"name": name, "reason": "Left before the window."} for name in others],
        "disagreement_with_engine": disagreement,
    }


def test_agent_loop_rejects_fabrication_then_accepts_correction(api):
    script = [
        message(
            [
                {"type": "text", "text": "Searching."},
                tool_use(1, "search_evidence", {"query": "Sam Whitlock badge"}),
                tool_use(2, "read_document", {"doc_id": "D8"}),
            ],
            "tool_use",
        ),
        message(
            [
                tool_use(
                    3,
                    "record_finding",
                    {
                        "statement": "Sam confessed.",
                        "status": "fact",
                        "citations": [{"passage_id": "D5:p1", "quote": "I confess to taking it"}],
                        "rationale": "",
                    },
                )
            ],
            "tool_use",
        ),
        message(
            [
                tool_use(
                    4,
                    "record_finding",
                    {
                        "statement": "Sam returned at 19:42.",
                        "status": "fact",
                        "citations": [{"passage_id": "D8:p2", "quote": "Sam Whitlock enters"}],
                        "rationale": "CCTV.",
                    },
                )
            ],
            "tool_use",
        ),
        message([tool_use(5, "submit_conclusion", conclusion(disagreement="The engine saw no extractions."))], "tool_use"),
    ]
    stub = api(script)
    case = case_one()
    inv = Investigation(case, EmptyExtractor())
    result = ClaudeReasoner(stub.gateway()).investigate(inv, auto_leads=False)

    assert result.culprit == "Sam Whitlock" and result.confidence == Confidence.MEDIUM
    assert [f.statement for f in inv.findings] == ["Sam returned at 19:42."]
    assert len(inv.finding_rejections) == 1

    rejection_turn = stub.requests[2]["messages"][-1]["content"][0]
    assert rejection_turn["is_error"] is True and "rejected" in rejection_turn["content"]
    tools = stub.requests[0]["tools"]
    assert all(tool["strict"] is True for tool in tools)
    assert "uncover_lead" not in {tool["name"] for tool in tools}
    assert stub.requests[0]["cache_control"] == {"type": "ephemeral"}
    assert stub.requests[0]["thinking"] == {"type": "adaptive"}


def test_agent_loop_retries_when_the_model_stops_without_concluding(api):
    finding = {
        "statement": "Sam returned.",
        "status": "fact",
        "citations": [{"passage_id": "D8:p2", "quote": "Sam Whitlock enters"}],
        "rationale": "",
    }
    stub = api(
        [
            message([{"type": "text", "text": "I think I am done."}]),
            message([tool_use(1, "record_finding", finding)], "tool_use"),
            message([tool_use(2, "submit_conclusion", conclusion(disagreement="The engine saw no extractions."))], "tool_use"),
        ]
    )
    inv = Investigation(case_one(), EmptyExtractor())
    assert ClaudeReasoner(stub.gateway()).investigate(inv).culprit == "Sam Whitlock"


def tools_for_case_one():
    inv = Investigation(case_one(), EmptyExtractor())
    inv.reassess("start")
    return InvestigatorTools(inv), inv


def add_fact(tools):
    tools.run(
        "record_finding",
        {
            "statement": "Sam returned.",
            "status": "fact",
            "citations": [{"passage_id": "D8:p2", "quote": "Sam Whitlock enters"}],
            "rationale": "",
        },
    )


def test_conclusion_must_name_a_real_suspect():
    tools, _ = tools_for_case_one()
    add_fact(tools)
    with pytest.raises(ToolError, match="not a suspect"):
        tools.run("submit_conclusion", conclusion(culprit="The Butler"))


def test_conclusion_cannot_rest_on_speculation():
    tools, _ = tools_for_case_one()
    tools.run("record_finding", {"statement": "Maybe Sam.", "status": "speculation", "citations": [], "rationale": ""})
    with pytest.raises(ToolError, match="speculation"):
        tools.run("submit_conclusion", conclusion())


def test_conclusion_must_account_for_every_suspect():
    tools, _ = tools_for_case_one()
    add_fact(tools)
    incomplete = conclusion() | {"eliminated": []}
    with pytest.raises(ToolError, match="ruled out"):
        tools.run("submit_conclusion", incomplete)


def test_disagreeing_with_the_engine_needs_a_stated_reason():
    tools, inv = tools_for_case_one()
    add_fact(tools)
    leader = inv.rounds[-1].leader
    culprit = next(p.name for p in inv.case.suspects if p.name != leader)
    with pytest.raises(ToolError, match="rule engine ranks"):
        tools.run("submit_conclusion", conclusion(culprit=culprit))
    tools.run("submit_conclusion", conclusion(culprit=culprit, disagreement="Extraction missed the lobby footage."))
    assert inv.conclusion.engine_agrees is False
    assert "lobby footage" in inv.conclusion.engine_note


def test_semantic_contradiction_needs_two_documents():
    tools, inv = tools_for_case_one()
    one_doc = {
        "person": "Sam Whitlock",
        "summary": "x",
        "resolution": "y",
        "discredited_speaker": "",
        "citations": [{"passage_id": "D5:p1", "quote": "I left the studio"}],
    }
    with pytest.raises(ToolError, match="two different documents"):
        tools.run("record_contradiction", one_doc)
    two_docs = one_doc | {
        "citations": [
            {"passage_id": "D5:p2", "quote": "I did not come back"},
            {"passage_id": "D8:p2", "quote": "Sam Whitlock enters"},
        ]
    }
    tools.run("record_contradiction", two_docs)
    assert inv.semantic_contradictions[0].kind == "semantic"


def test_read_tools_return_json_on_a_real_analysis():
    case = case_one()
    inv = Investigation(case, HeuristicExtractor(case))
    RuleReasoner().investigate(inv)
    tools = InvestigatorTools(inv)
    for name, arguments in [
        ("search_evidence", {"query": "laptop sleeve"}),
        ("read_document", {"doc_id": "D2"}),
        ("get_timeline", {"person": "Sam Whitlock"}),
        ("get_contradictions", {}),
        ("get_suspect_matrix", {}),
    ]:
        assert json.loads(tools.run(name, arguments))


def test_uncover_tool_returns_the_new_document():
    tools, inv = tools_for_case_one()
    result = json.loads(tools.run("uncover_lead", {"lead_id": "L1", "reason": "Find the serial number."}))
    assert result["document"]["id"] == "S1" and inv.uncovered == ["L1"]
    with pytest.raises(ToolError):
        tools.run("uncover_lead", {"lead_id": "L1", "reason": "Again."})


def test_unknown_tool_is_an_error():
    tools, _ = tools_for_case_one()
    with pytest.raises(ToolError):
        tools.run("delete_evidence", {})


def test_tool_schemas_are_strict_compatible():
    tools, _ = tools_for_case_one()
    for tool in tools.definitions():
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_answer_drops_bad_citations(api):
    draft = {
        "answer": "Sam returned.",
        "status": "fact",
        "citations": [
            {"passage_id": "D8:p2", "quote": "Sam Whitlock enters"},
            {"passage_id": "D8:p2", "quote": "Sam was laughing"},
        ],
    }
    stub = api([message([{"type": "text", "text": json.dumps(draft)}])])
    inv = Investigation(case_one(), EmptyExtractor())
    answer = ClaudeReasoner(stub.gateway()).answer(inv, "Who came back?")
    assert len(answer.citations) == 1 and answer.grounded is False
    assert answer.status == Status.FACT


def test_answer_with_only_bad_citations_is_withheld(api):
    draft = {"answer": "The butler did it.", "status": "fact", "citations": [{"passage_id": "D8:p2", "quote": "the butler"}]}
    stub = api([message([{"type": "text", "text": json.dumps(draft)}])])
    inv = Investigation(case_one(), EmptyExtractor())
    answer = ClaudeReasoner(stub.gateway()).answer(inv, "Who did it?")
    assert "butler" not in answer.answer and not answer.citations
