"""Smoke-test Tinker's hosted OpenAI-compatible API without running a benchmark.

Accepts a base model identifier or a saved sampler checkpoint as model_name.
Uses at most four short generations: text, tool call, tool result, and streaming.
No sandbox is created and no model weights are updated.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import chz
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


@chz.chz
class Config:
    model_name: str
    base_url: str = BASE_URL
    env_file: str = ".env"
    output_path: str = "notes/experiments/tinker-openai-compatible"
    thinking_effort: float = 0.9
    max_tokens: int = 512


async def main(config: Config) -> None:
    load_env_file(Path(config.env_file))
    root = Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(chz.asdict(config), indent=2))
    print(f"Results: {root.resolve()}", flush=True)
    extra_body = {"reasoning_effort": config.thinking_effort, "separate_reasoning": True}

    # Disable automatic retries for this diagnostic so each capability is
    # attempted once. Do not impose a client-side sampling latency deadline.
    async with AsyncOpenAI(
        api_key=os.environ["TINKER_API_KEY"],
        base_url=config.base_url,
        timeout=None,
        max_retries=0,
    ) as client:

        async def text_probe() -> dict[str, object]:
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": "Reply with exactly TINKER_OK."}],
                max_tokens=config.max_tokens,
                temperature=1.0,
                extra_body=extra_body,
            )
            return {
                "passed": (response.choices[0].message.content or "").strip() == "TINKER_OK",
                "response": response.model_dump(),
            }

        async def tool_probe() -> dict[str, object]:
            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "user",
                    "content": "Call lookup_probe once, then reply with exactly the value it returns.",
                }
            ]
            tools: list[ChatCompletionToolParam] = [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_probe",
                        "description": "Return the current test value.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ]
            first = await client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                tools=tools,
                max_tokens=config.max_tokens,
                temperature=1.0,
                extra_body=extra_body,
            )
            message = first.choices[0].message
            calls = message.tool_calls or []
            if len(calls) != 1 or calls[0].type != "function":
                return {"passed": False, "first_response": first.model_dump()}
            call = calls[0]
            if call.function.name != "lookup_probe" or json.loads(call.function.arguments) != {}:
                return {"passed": False, "first_response": first.model_dump()}
            expected = "probe-" + uuid4().hex[:12]
            # Preserve the returned assistant message, including any reasoning
            # extension, instead of reconstructing its tool call as plain text.
            messages.append(cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True)))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": expected})
            second = await client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                tools=tools,
                max_tokens=config.max_tokens,
                temperature=1.0,
                extra_body=extra_body,
            )
            return {
                "passed": (second.choices[0].message.content or "").strip() == expected,
                "expected": expected,
                "first_response": first.model_dump(),
                "second_response": second.model_dump(),
            }

        async def stream_probe() -> dict[str, object]:
            stream = await client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": "Reply with exactly STREAM_OK."}],
                max_tokens=config.max_tokens,
                temperature=1.0,
                extra_body=extra_body,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = []
            answer = ""
            async for chunk in stream:
                chunks.append(chunk.model_dump())
                for choice in chunk.choices:
                    answer += choice.delta.content or ""
            return {"passed": answer.strip() == "STREAM_OK", "answer": answer, "chunks": chunks}

        async def run(name: str, probe: Callable[[], Awaitable[dict[str, object]]]) -> bool:
            started = time.monotonic()
            try:
                result = await probe()
            except Exception as error:
                result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            result["seconds"] = round(time.monotonic() - started, 2)
            (root / f"{name}.json").write_text(json.dumps(result, indent=2))
            print(
                name,
                json.dumps(
                    {k: v for k, v in result.items() if k in {"passed", "seconds", "error"}}
                ),
                flush=True,
            )
            return result["passed"] is True

        if not await run("chat", text_probe):
            raise RuntimeError("Basic chat probe failed; inspect chat.json before proceeding")
        passed = await asyncio.gather(
            run("tool_roundtrip", tool_probe), run("stream", stream_probe)
        )
        if not all(passed):
            raise RuntimeError("A compatibility probe failed; inspect saved results")


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
