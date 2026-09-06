"""Hosted Chat Completions transport for mining and evaluation (not RL data)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage

from tinker_cookbook.renderers.base import Message, message_to_jsonable

TINKER_CHAT_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def create_chat_client(base_url: str, api_key_env: str = "TINKER_API_KEY") -> AsyncOpenAI:
    key = os.environ.get(api_key_env)
    if not key:
        raise ValueError(f"{api_key_env} is required")
    return AsyncOpenAI(base_url=base_url, api_key=key, timeout=None)


@dataclass
class FunctionCall:
    call_id: str
    name: str
    arguments: str
    type: str = "function_call"


@dataclass
class ChatTurn:
    id: str
    output_text: str
    output: list[FunctionCall]
    usage: CompletionUsage | None
    assistant_message: dict[str, object]
    finish_reason: str


class ChatSession:
    """One task's conversation; never share a session across concurrent tasks."""

    def __init__(
        self, client: AsyncOpenAI, model: str, effort: float, temperature: float = 1.0
    ) -> None:
        if not math.isfinite(effort) or not 0 <= effort <= 0.99:
            raise ValueError("Chat reasoning effort must be finite and in [0, 0.99]")
        self.client, self.model, self.effort, self.temperature = client, model, effort, temperature
        self.messages: list[ChatCompletionMessageParam] = []

    async def create(
        self,
        inputs: list[dict[str, object]],
        *,
        instructions: str,
        tools: list[dict[str, object]],
        max_tokens: int,
    ) -> ChatTurn:
        if not self.messages:
            self.messages.append({"role": "system", "content": instructions})
        for item in inputs:
            if item.get("type") == "function_call_output":
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(item["call_id"]),
                        "content": str(item["output"]),
                    }
                )
            else:
                self.messages.append(cast(ChatCompletionMessageParam, item))
        chat_tools = cast(
            list[ChatCompletionToolParam],
            [
                {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
                for tool in tools
            ],
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=chat_tools,
            max_tokens=max_tokens,
            temperature=self.temperature,
            extra_body={"reasoning_effort": self.effort, "separate_reasoning": True},
        )
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        self.messages.append(cast(ChatCompletionMessageParam, message))
        calls = []
        for call in choice.message.tool_calls or []:
            if call.type != "function":
                raise ValueError("Unsupported non-function tool call")
            calls.append(FunctionCall(call.id, call.function.name, call.function.arguments))
        return ChatTurn(
            response.id,
            choice.message.content or "",
            calls,
            response.usage,
            message,
            choice.finish_reason,
        )


class ChatAnnotationCompleter:
    """Stateless message completer with per-request provenance and usage evidence."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        thinking_effort: float,
        reports_dir: Path,
        base_url: str = TINKER_CHAT_BASE_URL,
        api_key_env: str = "TINKER_API_KEY",
        temperature: float = 1.0,
    ) -> None:
        if not math.isfinite(thinking_effort) or not 0 <= thinking_effort <= 0.99:
            raise ValueError("Chat reasoning effort must be finite and in [0, 0.99]")
        self.client = create_chat_client(base_url, api_key_env)
        self.model, self.max_tokens = model, max_tokens
        self.effort, self.temperature = thinking_effort, temperature
        self.reports_dir = reports_dir

    async def __call__(self, messages: list[Message]) -> Message:
        payload = [message_to_jsonable(m) for m in messages]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], payload),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body={"reasoning_effort": self.effort, "separate_reasoning": True},
        )
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / f"{uuid4().hex}.json").write_text(
            json.dumps(
                {
                    "request_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest(),
                    "model": self.model,
                    "base_url": str(self.client.base_url),
                    "thinking_effort": self.effort,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "response": response.model_dump(),
                },
                indent=2,
            )
        )
        choice = response.choices[0]
        if (
            choice.finish_reason != "stop"
            or choice.message.tool_calls
            or not choice.message.content
        ):
            raise ValueError(f"Incomplete annotation response: {choice.finish_reason}")
        return {"role": "assistant", "content": choice.message.content}

    async def close(self) -> None:
        await self.client.close()


def prepare_annotation_state(reports_dir: Path, config: dict[str, object]) -> None:
    """Fail closed when cached annotations came from different inference settings."""
    path = reports_dir / "inference_config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Annotation configuration differs; use a new reports directory")
    elif any(reports_dir.glob("*.json")):
        raise ValueError("Legacy reports lack inference provenance; use a new reports directory")
    reports_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True))
