import asyncio
import copy
import fcntl
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
from tinker_cookbook.recipes.harbor_rl.harbor_env import HARBOR_SYSTEM_PROMPT
from tinker_cookbook.recipes.kokkos_rl.chat_inference_test import completion
from tinker_cookbook.recipes.kokkos_rl.rl import eval_kokkos_openai as evaluation
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_recovery import (
    PrefixReplayTransport,
    ReadOnlyPrefix,
    recovery_accounting,
    validate_recovery_source,
)


def transcript():
    first = completion(calls=True)
    return json.dumps(
        [
            {
                "turn": 1,
                "response_id": "original-response",
                "finish_reason": "tool_calls",
                "assistant_message": first["choices"][0]["message"],
                "raw_usage": first["usage"],
                "tool_outputs": [
                    {
                        "call_id": "call_1",
                        "output": "original observation",
                        "model_output_truncated": False,
                    }
                ],
            }
        ]
    )


def original_result():
    return evaluation.OpenAITaskResult(
        "task",
        0,
        {},
        1,
        1,
        20,
        3,
        None,
        600,
        "error",
        cached_input_tokens=5,
        estimated_cost_usd=0.000026,
        error="NotFoundError: Error code: 404 - model does not exist",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "observations_match,capture_fails", [(True, False), (True, True), (False, False)]
)
async def test_recovery_uses_original_harness_and_never_resamples_prefix(
    tmp_path, monkeypatch, observations_match, capture_fails
):
    task = make_task(tmp_path, "task")
    config = evaluation.CLIConfig(
        model_name="test",
        api_mode="chat",
        chat_provider="nebius",
        reasoning_effort="high",
        temperature=None,
        max_turns=2,
        max_tokens=10,
        max_sampled_tokens=12,
        input_price_per_million=1,
        cached_input_price_per_million=1,
        output_price_per_million=2,
        max_cost_usd_per_task=None,
    )
    first_request = {
        "model": "test",
        "messages": [
            {"role": "system", "content": HARBOR_SYSTEM_PROMPT},
            {"role": "user", "content": task.instruction},
        ],
        "max_tokens": 10,
        "reasoning_effort": "high",
        "tools": [
            {
                "type": "function",
                "function": {k: v for k, v in evaluation.BASH_TOOL.items() if k != "type"},
            }
        ],
    }
    prefix = ReadOnlyPrefix.from_transcript(transcript(), approved_commands=("pwd",))
    original = original_result()
    validate_recovery_source(original, prefix)
    requests = []

    def live_provider(request):
        requests.append(json.loads(request.content))
        response = completion()
        response["id"] = "new-live-response"
        return httpx.Response(200, json=response)

    transport = PrefixReplayTransport(
        prefix=prefix,
        expected_first_request=first_request,
        transport=httpx.MockTransport(live_provider),
    )
    sandbox = SimpleNamespace(cleanup=AsyncMock())
    monkeypatch.setattr(
        evaluation,
        "HarborBashTool",
        lambda *a, **kw: SimpleNamespace(
            bash=SimpleNamespace(
                run=AsyncMock(
                    return_value=SimpleNamespace(
                        messages=[
                            {
                                "content": "original observation"
                                if observations_match
                                else "CHANGED observation",
                            }
                        ]
                    )
                )
            ),
        ),
    )
    monkeypatch.setattr(evaluation, "HarborReward", lambda **kw: AsyncMock(return_value=(1, {})))
    destination = tmp_path / "recovery"
    destination.mkdir()
    capture = AsyncMock(side_effect=RuntimeError("artifact unavailable") if capture_fails else None)
    async with AsyncOpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        # OpenAI accepts legacy httpx clients through its compatibility adapter.
        http_client=cast(DefaultAsyncHttpxClient, httpx.AsyncClient(transport=transport)),
    ) as client:
        result = await evaluation.evaluate_task(
            task,
            client,
            AsyncMock(return_value=sandbox),
            config,
            destination,
            asyncio.Lock(),
            before_grading=capture,
        )
    assert transport.replayed
    if observations_match:
        assert result.error is None
        assert result.reward == 1
        capture.assert_awaited_once_with(sandbox)
        assert result.turns_used == 2
        assert result.input_tokens == 40 and result.output_tokens == 6
        assert len(requests) == 1
        assert requests[0]["max_tokens"] == 9
        assert requests[0]["messages"][2]["reasoning_content"] == "retained reasoning"
        assert transport.observations_verified
        accounting = recovery_accounting(original, result, prefix_replayed=True)
        assert accounting["new_billable_usage"] == {
            "input_tokens": 20,
            "output_tokens": 3,
            "cached_input_tokens": 5,
            "cache_write_input_tokens": 0,
            "estimated_cost_usd": 0.000026,
        }
    else:
        assert result.error is not None
        capture.assert_not_awaited()
        assert requests == []
        assert not transport.observations_verified
        usage = recovery_accounting(original, result, prefix_replayed=True)["new_billable_usage"]
        assert isinstance(usage, dict) and usage["output_tokens"] == 0
    saved = json.loads((destination / "task.json").read_text())
    assert saved[0]["response_id"] == "original-response"
    sandbox.cleanup.assert_awaited_once()


def test_recovery_rejects_unapproved_or_incomplete_prefix_and_sample_discard():
    with pytest.raises(ValueError, match="approved"):
        ReadOnlyPrefix.from_transcript(transcript(), approved_commands=("different command",))
    incomplete = json.loads(transcript())
    incomplete[0]["tool_outputs"] = []
    with pytest.raises(ValueError, match="complete observation"):
        ReadOnlyPrefix.from_transcript(json.dumps(incomplete), approved_commands=("pwd",))
    with pytest.raises(ValueError, match="discard"):
        validate_recovery_source(original_result(), None)
    source = copy.copy(original_result())
    source.turns_used = source.tool_calls = source.input_tokens = source.output_tokens = 0
    validate_recovery_source(source, None)
    source.error = None
    with pytest.raises(ValueError, match="missing-model"):
        validate_recovery_source(source, None)


def test_accounting_before_prefix_replay_does_not_subtract_unreplayed_usage():
    original = original_result()
    failed_before_replay = replace(
        original,
        turns_used=0,
        tool_calls=0,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        estimated_cost_usd=0,
    )
    usage = recovery_accounting(original, failed_before_replay, prefix_replayed=False)
    assert usage["new_billable_usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "estimated_cost_usd": 0,
    }
    assert usage["original_attempt_usage"] != usage["new_billable_usage"]


@pytest.mark.asyncio
async def test_recovery_cannot_start_while_original_controller_owns_slots(tmp_path, monkeypatch):
    from tinker_cookbook.recipes.kokkos_rl.rl import recover_nebius_task as runner

    recover = AsyncMock()
    monkeypatch.setattr(runner, "recover", recover)
    config = runner.RecoveryConfig(source_root=str(tmp_path), model="original", task_name="task")
    with (tmp_path / "controller.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="shared inference slots"):
            await runner.main(config)
        recover.assert_not_awaited()
    await runner.main(config)
    recover.assert_awaited_once_with(config)
