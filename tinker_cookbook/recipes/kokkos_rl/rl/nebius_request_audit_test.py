import asyncio
import json

import pytest
from openai.types.chat import ChatCompletion

from tinker_cookbook.recipes.kokkos_rl.rl.nebius_request_audit import AuditedCompletions


def completion(*, usage: bool = True) -> ChatCompletion:
    value = {
        "id": "response-1",
        "created": 1,
        "model": "exact-model",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
            }
        ],
    }
    if usage:
        value["usage"] = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    return ChatCompletion.model_validate(value)


@pytest.mark.asyncio
async def test_persist_request_before_dispatch_and_provider_usage_before_return(tmp_path):
    directory = tmp_path / "requests"

    async def provider(**kwargs: object) -> ChatCompletion:
        saved = json.loads((directory / "001/request.json").read_text())
        assert saved["arguments"] == kwargs
        assert saved["automatic_retries"] == 0
        assert (directory / "001/dispatch_started.json").exists()
        return completion()

    audited = AuditedCompletions(provider, directory, "identity")
    response = await audited.create(
        model="exact-model", messages=[{"role": "user", "content": "p"}]
    )
    assert response.id == "response-1"
    assert json.loads((directory / "001/response.json").read_text())["usage"]["total_tokens"] == 10
    assert json.loads((directory / "001/completion.json").read_text())["usage_available"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ConnectionError, asyncio.CancelledError])
async def test_uncertain_request_is_recorded_once_and_never_retried(tmp_path, failure):
    calls = 0

    async def provider(**kwargs: object) -> ChatCompletion:
        nonlocal calls
        calls += 1
        raise failure("sensitive error body is not persisted")

    audited = AuditedCompletions(provider, tmp_path / "requests", "identity")
    with pytest.raises(failure):
        await audited.create(model="exact-model")
    assert calls == 1
    error = json.loads(
        (tmp_path / "requests/001/unreceived_or_unpersisted_response.json").read_text()
    )
    assert error["usage"] == "unknown"
    assert error["retry_permitted"] is False
    assert "sensitive" not in json.dumps(error)
    with pytest.raises(ValueError, match="Prior generation"):
        AuditedCompletions(provider, tmp_path / "requests", "identity")


@pytest.mark.asyncio
async def test_missing_usage_preserves_raw_response_without_inventing_zero(tmp_path):
    async def provider(**kwargs: object) -> ChatCompletion:
        return completion(usage=False)

    audited = AuditedCompletions(provider, tmp_path / "requests", "identity")
    await audited.create(model="exact-model")
    saved = json.loads((tmp_path / "requests/001/completion.json").read_text())
    assert saved["usage"] is None
    assert saved["usage_available"] is False
    assert (tmp_path / "requests/001/response.json").exists()


@pytest.mark.asyncio
async def test_disk_claim_collision_and_streaming_fail_before_dispatch(tmp_path):
    calls = 0

    async def provider(**kwargs: object) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return completion()

    directory = tmp_path / "requests"
    audited = AuditedCompletions(provider, directory, "identity")
    with pytest.raises(ValueError, match="non-stream"):
        await audited.create(stream=True)
    assert not directory.exists()
    (directory / "001").mkdir(parents=True)
    (directory / "001/request.json").write_text("partial")
    with pytest.raises(FileExistsError):
        await audited.create(model="exact-model")
    assert calls == 0
    assert (directory / "001/request.json").read_text() == "partial"


@pytest.mark.asyncio
async def test_real_chat_session_omit_and_reasoning_tool_roundtrip(tmp_path):
    from types import SimpleNamespace

    from openai import Omit

    from tinker_cookbook.recipes.kokkos_rl.chat_inference import ChatSession

    calls = []

    async def provider(**kwargs: object) -> ChatCompletion:
        assert isinstance(kwargs["temperature"], Omit)
        calls.append(kwargs)
        if len(calls) == 1:
            response = completion().model_dump()
            response["choices"][0] = {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "kept reasoning",
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                        }
                    ],
                },
            }
            return ChatCompletion.model_validate(response)
        return completion()

    audited = AuditedCompletions(provider, tmp_path / "requests", "identity")
    client = SimpleNamespace(chat=SimpleNamespace(completions=audited))
    session = ChatSession(
        client, "exact-model", 0.9, None, provider="nebius", reasoning_effort="high"
    )
    tools = [
        {
            "type": "function",
            "name": "bash",
            "description": "shell",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        }
    ]
    await session.create(
        [{"role": "user", "content": "task"}], instructions="system", tools=tools, max_tokens=64
    )
    await session.create(
        [{"type": "function_call_output", "call_id": "call1", "output": "/workspace/repo"}],
        instructions="system",
        tools=tools,
        max_tokens=64,
    )
    first = json.loads((tmp_path / "requests/001/request.json").read_text())
    second = json.loads((tmp_path / "requests/002/request.json").read_text())
    assert len(calls) == 2
    assert first["omitted_sdk_arguments"] == ["temperature"]
    assert "temperature" not in first["arguments"]
    assert first["arguments"]["extra_body"] == {"reasoning_effort": "high"}
    assert second["arguments"]["messages"][-2]["reasoning_content"] == "kept reasoning"
    assert second["arguments"]["messages"][-1]["tool_call_id"] == "call1"
    assert (tmp_path / "requests/002/response.json").exists()


@pytest.mark.asyncio
async def test_unserializable_local_argument_leaves_no_request_and_never_calls_provider(tmp_path):
    calls = 0

    async def provider(**kwargs: object) -> ChatCompletion:
        nonlocal calls
        calls += 1
        return completion()

    audited = AuditedCompletions(provider, tmp_path / "requests", "identity")
    with pytest.raises(TypeError):
        await audited.create(unserializable=object())
    assert calls == 0
    assert not (tmp_path / "requests").exists()
