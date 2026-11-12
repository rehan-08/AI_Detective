"""Groq as an alternative model provider.

GroqGateway offers the same parse() and turn() as ClaudeGateway, so the extraction and
agent code, and every guard around them, are shared. The conversation is kept in the
Anthropic block format and translated at this boundary only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import groq
from pydantic import BaseModel, ValidationError

from .claude_backend import MAX_TOKENS, ClaudeExtractor, ClaudeReasoner, RefusedError

DEFAULT_MODEL = "openai/gpt-oss-120b"  # supports both tool use and strict structured outputs
MAX_ATTEMPTS = 3


def groq_model() -> str:
    return os.environ.get("GROQ_MODEL") or DEFAULT_MODEL


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"


class GroqGateway:
    def __init__(self, client: groq.Groq | None = None, model: str | None = None):
        self.client = client or groq.Groq()
        self.model = model or groq_model()
        self.input_tokens = 0
        self.output_tokens = 0

    def parse(self, system: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": strict_schema(schema)},
            },
        )
        choice = self._check(response)
        try:
            return schema.model_validate_json(choice.message.content or "")
        except ValidationError as error:
            raise RuntimeError(f"Groq returned output that does not match the schema: {error.errors()[0]['msg']}") from error

    def turn(self, system: str, messages: list[dict], tools: list[dict]) -> SimpleNamespace:
        request = {
            "model": self.model,
            "max_completion_tokens": MAX_TOKENS,
            "messages": to_chat_messages(system, messages),
            "tools": [to_chat_tool(tool) for tool in tools],
        }
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                choice = self._check(self.client.chat.completions.create(**request))
                return SimpleNamespace(content=from_chat_message(choice.message))
            except groq.BadRequestError as error:
                # Groq rejects a generated tool call it cannot parse; a new attempt usually succeeds.
                if "tool_use_failed" not in str(error) or attempt == MAX_ATTEMPTS:
                    raise
        raise AssertionError("unreachable")

    def _check(self, response):
        if response.usage:
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise RefusedError("Groq's content filter stopped the response.")
        if choice.finish_reason == "length":
            raise RuntimeError(f"response truncated at {MAX_TOKENS} tokens")
        return choice


def strict_schema(schema: type[BaseModel]) -> dict:
    """Inline $ref definitions, so the schema stands alone."""
    document = schema.model_json_schema()
    definitions = document.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(definitions[node["$ref"].rsplit("/", 1)[-1]])
            return {key: inline(value) for key, value in node.items()}
        if isinstance(node, list):
            return [inline(item) for item in node]
        return node

    return inline(document)


def to_chat_tool(tool: dict) -> dict:
    function = {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}
    return {"type": "function", "function": function}


def to_chat_messages(system: str, messages: list[dict]) -> list[dict]:
    chat = [{"role": "system", "content": system}]
    names: dict[str, str] = {}
    for item in messages:
        content = item["content"]
        if isinstance(content, str):
            chat.append({"role": item["role"], "content": content})
        elif item["role"] == "assistant":
            chat.append(_assistant_message(content, names))
        else:
            chat.extend(
                {"role": "tool", "tool_call_id": b["tool_use_id"], "name": names.get(b["tool_use_id"], ""), "content": b["content"]}
                for b in content
            )
    return chat


def _assistant_message(blocks: list, names: dict[str, str]) -> dict:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(b.text for b in blocks if b.type == "text") or None,
    }
    calls = [b for b in blocks if b.type == "tool_use"]
    if calls:
        names.update({call.id: call.name for call in calls})
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.input)}}
            for call in calls
        ]
    return message


def from_chat_message(message) -> list:
    blocks: list = [TextBlock(text=message.content)] if message.content else []
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"invalid_json": call.function.arguments}  # the tool layer reports this to the model
        blocks.append(ToolUseBlock(id=call.id, name=call.function.name, input=arguments))
    return blocks


class GroqExtractor(ClaudeExtractor):
    name = "groq"
    workers = 2  # lower than Claude's, to stay inside Groq's rate limits


class GroqReasoner(ClaudeReasoner):
    name = "groq"
