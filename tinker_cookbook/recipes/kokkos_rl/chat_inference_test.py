import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
from tinker_cookbook.recipes.kokkos_rl import chat_inference
from tinker_cookbook.recipes.kokkos_rl.chat_inference import (
    ChatAnnotationCompleter,
    ChatSession,
    prepare_annotation_state,
)
from tinker_cookbook.recipes.kokkos_rl.rl import eval_kokkos_openai as evaluation
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_tinker_chat import TinkerChatConfig


def completion(*, calls=False, finish=None, usage=True, content="done"):
    message = {"role": "assistant", "content": content, "reasoning_content": "retained reasoning"}
    if calls:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"pwd"}',
                },
            }
        ]
    result = {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish or ("tool_calls" if calls else "stop"),
            }
        ],
    }
    if usage:
        result["usage"] = {
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "total_tokens": 23,
            "prompt_tokens_details": {"cached_tokens": 5},
        }
    return result


def api_client(handler):
    return AsyncOpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_tool_history_reasoning_and_concurrent_task_isolation():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(copy.deepcopy(payload))
        return httpx.Response(200, json=completion(calls=payload["messages"][-1]["role"] != "tool"))

    async with api_client(handler) as client:
        sessions = [ChatSession(client, name, 0.9) for name in ("a", "b")]
        for index in range(2):
            await asyncio.gather(
                *[
                    session.create(
                        [{"role": "user", "content": f"task {session.model}"}]
                        if index == 0
                        else [
                            {
                                "type": "function_call_output",
                                "call_id": "call_1",
                                "output": f"/{session.model}",
                            }
                        ],
                        instructions="system",
                        tools=[evaluation.BASH_TOOL],
                        max_tokens=32,
                    )
                    for session in sessions
                ]
            )
    for request in requests:
        assert request["reasoning_effort"] == 0.9
        assert request["temperature"] == 1
        assert request["tools"][0]["function"]["name"] == "bash"
        assert request["messages"][1]["content"] == f"task {request['model']}"
        assert "previous_response_id" not in request
        if len(request["messages"]) == 4:
            assert request["messages"][2]["reasoning_content"] == "retained reasoning"
            assert request["messages"][3] == {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": f"/{request['model']}",
            }


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["stop", "length"])
async def test_annotation_records_usage_and_rejects_truncation(tmp_path, monkeypatch, finish):
    client = api_client(
        lambda request: httpx.Response(200, json=completion(finish=finish, content='{"ok":true}'))
    )
    monkeypatch.setattr(chat_inference, "create_chat_client", lambda *a: client)
    completer = ChatAnnotationCompleter(
        model="test", max_tokens=20, thinking_effort=0.9, reports_dir=tmp_path
    )
    try:
        if finish == "length":
            with pytest.raises(ValueError, match="Incomplete annotation"):
                await completer([{"role": "user", "content": "annotate"}])
        else:
            assert await completer([{"role": "user", "content": "annotate"}]) == {
                "role": "assistant",
                "content": '{"ok":true}',
            }
    finally:
        await completer.close()
    records = list(tmp_path.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["response"]["usage"]["completion_tokens"] == 3
    assert record["response"]["choices"][0]["message"]["reasoning_content"] == "retained reasoning"
    assert "api_key" not in record


def test_annotation_resume_checks_provider_model_effort_and_inputs(tmp_path):
    config = {"provider": "tinker-chat", "model": "a", "effort": 0.9, "input_sha256": "x"}
    prepare_annotation_state(tmp_path, config)
    prepare_annotation_state(tmp_path, config)
    for key, value in (
        ("provider", "tinker"),
        ("model", "b"),
        ("effort", 0.7),
        ("input_sha256", "y"),
    ):
        with pytest.raises(ValueError, match="configuration differs"):
            prepare_annotation_state(tmp_path, {**config, key: value})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,finish,expected",
    [(2, "stop", "model_finished"), (1, "stop", "max_turns"), (2, "length", "max_tokens")],
)
async def test_eval_chat_end_to_end_budget_grading_and_unknown_usage(
    tmp_path, monkeypatch, limit, finish, expected
):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json=completion(
                calls=len(requests) == 1, finish=None if len(requests) == 1 else finish
            ),
        )

    client = api_client(handler)
    monkeypatch.setattr(evaluation, "create_chat_client", lambda *args: client)
    sandbox = SimpleNamespace(cleanup=AsyncMock())
    factory = AsyncMock(return_value=sandbox)
    bash = AsyncMock(return_value=SimpleNamespace(messages=[{"content": "a" * 200}]))
    monkeypatch.setattr(
        evaluation,
        "HarborBashTool",
        lambda *a, **kw: SimpleNamespace(bash=SimpleNamespace(run=bash)),
    )
    grader = AsyncMock(return_value=(1.0, {"pass": 1.0}))
    monkeypatch.setattr(evaluation, "HarborReward", lambda **kw: grader)
    task = make_task(tmp_path, "task")
    config = TinkerChatConfig(
        resume_dir=str(tmp_path / "results"), max_turns=limit, max_tool_output_chars=60
    )
    results = await evaluation.run_eval(config, [task], factory)
    result = results[0]
    assert result.error is None
    assert result.stop_reason == expected
    assert result.turns_used == limit
    assert result.reasoning_tokens is None
    assert result.estimated_cost_usd is None
    assert result.input_tokens == 20 * limit
    assert result.cached_input_tokens == 5 * limit
    assert result.reward == 1
    assert bash.await_count == 1
    assert grader.await_count == 1
    sandbox.cleanup.assert_awaited_once()
    if limit == 2:
        assert len(requests[1]["messages"][-1]["content"]) == 60
    summary = json.loads((tmp_path / "results/result.json").read_text())
    assert summary["estimated_cost_usd"] is None
    assert summary["reasoning_tokens"] is None
    transcript = json.loads((tmp_path / "results/task.json").read_text())
    assert transcript[0]["assistant_message"]["reasoning_content"] == "retained reasoning"
    with pytest.raises(ValueError, match="configuration differs"):
        await evaluation.run_eval(
            TinkerChatConfig(resume_dir=config.resume_dir, base_url="https://different.invalid"),
            [task],
            factory,
        )


@pytest.mark.asyncio
async def test_missing_usage_fails_closed_and_exact_remaining_token_budget():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion(usage=False))

    async with api_client(handler) as client:
        response = await ChatSession(client, "a", 0.9).create(
            [{"role": "user", "content": "hi"}],
            instructions="system",
            tools=[],
            max_tokens=1,
        )
    assert sent[0]["max_tokens"] == 1
    with pytest.raises(ValueError, match="omitted usage"):
        evaluation._usage(response)
