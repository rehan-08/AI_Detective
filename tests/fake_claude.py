"""A local stand-in for the Claude API, so the real SDK and agent loop run with no key.

Requests go over real HTTP through the real SDK, which checks request and response
shapes. The replies are scripted, so this exercises the plumbing and the guards.
It says nothing about how well Claude reasons; only tests/test_live.py can show that.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import anthropic

from detective.claude_backend import ClaudeGateway
from detective.heuristic import HeuristicExtractor
from detective.models import Case


def message(content, stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def text(body: str, stop_reason="end_turn"):
    return message([{"type": "text", "text": body}], stop_reason)


def tool_use(index, name, arguments):
    return {"type": "tool_use", "id": f"toolu_{index}", "name": name, "input": arguments}


def api_error(status: int, error_type: str = "api_error"):
    return status, {"type": "error", "error": {"type": error_type, "message": f"Simulated {status}"}}


def scripted(replies: list) -> Callable[[dict], object]:
    """Reply with each item in turn. Running out is a test failure in the server log."""
    queue = list(replies)
    return lambda _request: queue.pop(0)


class FakeApi:
    """respond(request) returns a message, or a (status, body) pair for an error."""

    def __init__(self, respond: Callable[[dict], object]):
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["content-length"])
                request = json.loads(self.rfile.read(length))
                fake.requests.append(request)
                fake.headers.append({k.lower(): v for k, v in self.headers.items()})
                reply = respond(request)
                status, payload = reply if isinstance(reply, tuple) else (200, reply)
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def gateway(self) -> ClaudeGateway:
        client = anthropic.Anthropic(api_key="fake-key-for-tests", base_url=self.url, max_retries=0)
        return ClaudeGateway(client=client)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def chat_reply(reply):
    """Translate a scripted Anthropic-format reply into Groq's chat format."""
    if isinstance(reply, tuple):
        return reply
    text_parts = [b["text"] for b in reply["content"] if b["type"] == "text"]
    calls = [
        {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
        for b in reply["content"] if b["type"] == "tool_use"
    ]
    finish = {"tool_use": "tool_calls", "max_tokens": "length", "end_turn": "stop"}.get(reply["stop_reason"], reply["stop_reason"])
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [{
            "index": 0,
            "finish_reason": finish,
            "message": {"role": "assistant", "content": "".join(text_parts) or None, "tool_calls": calls or None},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def on_groq(model):
    """Let a scripted model written for the Anthropic format serve Groq requests."""

    def respond(request):
        messages = [m for m in request["messages"] if m["role"] != "system"]
        trailing = []
        while messages and messages[-1]["role"] == "tool":
            trailing.insert(0, messages.pop())
        if trailing:
            messages.append({"role": "user", "content": [{"content": m["content"]} for m in trailing]})
        view = {"messages": messages, "tools": [{"name": t["function"]["name"]} for t in request.get("tools", [])]}
        if "response_format" in request:
            view["output_config"] = {"format": request["response_format"]}
        return chat_reply(model(view))

    return respond


def tool_results(request: dict) -> list[dict]:
    """The tool results in the latest user turn, decoded."""
    content = request["messages"][-1]["content"]
    if isinstance(content, str):
        return []
    return [json.loads(block["content"]) for block in content if not block.get("is_error")]


class DetectiveModel:
    """A well-behaved scripted model that plays through a whole investigation.

    It answers extraction requests with the offline extractor's output, follows every
    lead it is offered, then files the rule engine's leader with contradiction-based
    findings. The verdict therefore comes from the engine, not from any judgement here.
    """

    def __init__(self, case: Case):
        self.documents = {d.id: d for d in [*case.documents, *case.sealed.values()]}
        self.extractor = HeuristicExtractor(case)
        self.counter = 0
        self.stage = "start"
        self.pending: list[str] = []
        self.matrix: dict = {}

    def __call__(self, request: dict):
        if "format" in request.get("output_config", {}):
            return self._extract(request)
        if len(request["messages"]) == 1:
            allowed = {tool["name"] for tool in request["tools"]}
            brief = request["messages"][0]["content"]
            self.pending = re.findall(r"^- (L\d+):", brief, re.M) if "uncover_lead" in allowed else []
            self.stage = "start"
        return self._investigate(request)

    def _extract(self, request: dict):
        doc_id = re.search(r"^doc_id: (\S+)$", request["messages"][0]["content"], re.M).group(1)
        return text(self.extractor.extract(self.documents[doc_id]).model_dump_json())

    def _calls(self, *calls) -> dict:
        blocks = []
        for name, arguments in calls:
            self.counter += 1
            blocks.append(tool_use(self.counter, name, arguments))
        return message(blocks, "tool_use")

    def _investigate(self, request: dict):
        if self.stage == "start":
            self.stage = "leads"
            first = next(iter(self.documents))
            return self._calls(("search_evidence", {"query": "who was at the scene"}), ("read_document", {"doc_id": first}))
        if self.stage == "leads" and self.pending:
            return self._calls(("uncover_lead", {"lead_id": self.pending.pop(0), "reason": "It could change the conclusion."}))
        if self.stage == "leads":
            self.stage = "findings"
            return self._calls(("get_suspect_matrix", {}), ("get_timeline", {"person": ""}), ("get_contradictions", {}))
        if self.stage == "findings":
            self.stage = "conclude"
            matrix, _, contradictions = tool_results(request)
            self.matrix = matrix
            leader = matrix["assessment"]["leader"]
            against = [c for c in contradictions["contradictions"] if c["discredited_speaker"] == leader]
            return self._calls(*[("record_finding", self._finding(item)) for item in against])
        return self._calls(("submit_conclusion", self._conclusion(tool_results(request))))

    def _finding(self, contradiction: dict) -> dict:
        return {
            "statement": contradiction["summary"],
            "status": "fact",
            "citations": contradiction["citations"],
            "rationale": contradiction["resolution"],
        }

    def _conclusion(self, findings: list[dict]) -> dict:
        assessment = self.matrix["assessment"]
        leader = assessment["leader"]
        return {
            "culprit": leader,
            "accomplices": assessment["accomplices"],
            "confidence": assessment["confidence"],
            "confidence_explanation": assessment["confidence_explanation"],
            "narrative": f"The records point to {leader}.",
            "supporting_finding_ids": [f["id"] for f in findings],
            "eliminated": [
                {"name": s["name"], "reason": s["alibi_note"] or "Ranked lower."}
                for s in self.matrix["suspects"]
                if s["name"] != leader
            ],
            "disagreement_with_engine": "",
        }
