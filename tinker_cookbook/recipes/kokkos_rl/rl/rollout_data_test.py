"""Sampling and grading failures must retain evidence without another model call."""

import asyncio
import base64
import hashlib
import json
from unittest.mock import AsyncMock, Mock

import pytest
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.recipes.harbor_rl import eval as harbor_eval
from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
from tinker_cookbook.recipes.kokkos_rl.rl.rollout_data import RecordedTokenCompleter
from tinker_cookbook.sandbox import SandboxResult


@pytest.mark.asyncio
async def test_sampling_failure_keeps_request_and_never_retries(tmp_path):
    policy = AsyncMock(side_effect=RuntimeError("sampling interrupted"))
    recorded = RecordedTokenCompleter(policy, tmp_path)
    prompt = tinker.ModelInput.from_ints([11, 12])
    with pytest.raises(RuntimeError, match="sampling interrupted"):
        await recorded(prompt, [7], max_tokens=13)
    assert policy.await_count == 1
    assert recorded.calls_started == 1 and recorded.responses_received == 0
    assert json.loads((tmp_path / "000.request.json").read_text()) == {
        "input_tokens": [11, 12],
        "stop": [7],
        "max_tokens": 13,
    }
    assert not (tmp_path / "000.response.json").exists()
    # A replacement wrapper cannot silently overwrite the unresolved request.
    replacement = RecordedTokenCompleter(policy, tmp_path)
    with pytest.raises(FileExistsError):
        await replacement(prompt, [7])
    assert policy.await_count == 1


@pytest.mark.asyncio
async def test_grader_exception_preserves_raw_response_candidate_and_error_result(
    tmp_path, monkeypatch
):
    task = make_task(tmp_path, "task")
    (task.task_dir / "tests/test.sh").write_text("baseline=" + "a" * 40 + "\n")
    patch = b"candidate source patch"
    sandbox = AsyncMock()
    sandbox.sandbox_id = "fixture-sandbox"
    sandbox.run_command.return_value = SandboxResult(
        stdout=json.dumps(
            {
                "base_commit": "a" * 40,
                "head_commit": "a" * 40,
                "patch_bytes": len(patch),
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "patch_base64": base64.b64encode(patch).decode(),
            }
        ),
        stderr="",
        exit_code=0,
    )
    output = tmp_path / "results"
    output.mkdir()
    directory = output / "rollouts/task__00"
    history = [{"role": "assistant", "content": "finished"}]

    async def grade(messages):
        # Both must be on disk before any grader work can fail.
        assert json.loads((directory / "messages.json").read_text()) == history
        assert (directory / "candidate.patch").read_bytes() == patch
        raise RuntimeError("verifier produced no reward file")

    monkeypatch.setattr(harbor_eval, "HarborReward", Mock(return_value=grade))
    monkeypatch.setattr(harbor_eval, "_initial_messages", Mock(return_value=[]))
    monkeypatch.setattr(harbor_eval, "build_agent_tool_env", lambda **kwargs: kwargs["reward_fn"])
    prompt = tinker.ModelInput.from_ints([5, 6])
    response = TokensWithLogprobs([7, 8], [-0.1, -0.2], stop_reason="stop")
    policy = AsyncMock(return_value=response)

    async def rollout(recorded_policy, reward_fn):
        assert await recorded_policy(prompt, [9], max_tokens=17) is response
        await reward_fn(history)
        raise AssertionError("grader must fail")

    monkeypatch.setattr(harbor_eval, "do_single_rollout", rollout)
    result = await harbor_eval.evaluate_task(
        task,
        policy,
        Mock(),
        AsyncMock(return_value=sandbox),
        harbor_eval.EvalConfig(export_kokkos_rollouts=True),
        output,
        asyncio.Lock(),
    )
    assert result.error == "verifier produced no reward file"
    assert result.turns_used == 1
    assert result.reward_details == {
        "sampling/policy_calls_started": 1.0,
        "sampling/responses_received": 1.0,
    }
    policy.assert_awaited_once_with(prompt, [9], max_tokens=17)
    assert json.loads((directory / "sampling/000.response.json").read_text()) == {
        "tokens": [7, 8],
        "logprobs": [-0.1, -0.2],
        "stop_reason": "stop",
    }
    failure = json.loads((directory / "failure.json").read_text())
    assert failure["stage"] == "grading"
    assert failure["trajectory_complete"] is False
    assert failure["eligible_for_training"] is False
    assert not (directory / "trajectory.json").exists()
    sandbox.cleanup.assert_awaited_once()
