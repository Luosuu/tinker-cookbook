import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.recipes.kokkos_rl.rl import eval_kokkos_pi as module
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult


@pytest.mark.asyncio
async def test_bridge_dispatches_only_to_sandbox_and_journals(tmp_path: Path, monkeypatch):
    reader = asyncio.StreamReader()
    reader.feed_data((json.dumps({"type": "bash", "id": 1, "command": "pwd"}) + "\n").encode())
    reader.feed_data((json.dumps({"type": "result", "result": {"turns": 1}}) + "\n").encode())
    reader.feed_eof()

    class Input:
        def __init__(self):
            self.messages = []

        def write(self, data):
            self.messages.append(json.loads(data))

        async def drain(self):
            pass

    class Process:
        stdin = Input()
        stdout = reader
        returncode = 0

        async def wait(self):
            return 0

    process = Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    sandbox = AsyncMock()
    sandbox.run_command.return_value = SandboxResult(stdout="remote", stderr="", exit_code=0)
    result = await module.pi_rollout(
        module.CLIConfig(), HarborTask("task", "fix", tmp_path), sandbox, tmp_path
    )
    assert result["turns"] == 1
    assert process.stdin.messages[1]["result"]["stdout"] == "remote"
    sandbox.run_command.assert_awaited_once()
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 3


@pytest.mark.asyncio
async def test_candidate_precedes_verifier_and_cleanup(tmp_path: Path, monkeypatch):
    order = []
    sandbox = AsyncMock()

    async def rollout(*args):
        return {"turns": 2, "stop_reason": "max_turns"}

    async def capture(*args, **kwargs):
        order.append("capture")

    class Reward:
        def __init__(self, **kwargs):
            pass

        async def __call__(self, history):
            order.append("grade")
            return 1.0, {}

    monkeypatch.setattr(module, "pi_rollout", rollout)
    monkeypatch.setattr(module, "capture_candidate", capture)
    monkeypatch.setattr(module, "HarborReward", Reward)
    result = await module.evaluate_task(
        module.CLIConfig(),
        HarborTask("task", "fix", tmp_path),
        0,
        AsyncMock(return_value=sandbox),
        tmp_path / "trial",
    )
    assert order == ["capture", "grade"]
    assert result.reward == 1
    assert result.reward_details["natural_completion"] == 0
    sandbox.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_failure_is_error_not_scored_failure(tmp_path: Path, monkeypatch):
    sandbox = AsyncMock()
    monkeypatch.setattr(
        module, "pi_rollout", AsyncMock(side_effect=RuntimeError("provider failed"))
    )
    grader = AsyncMock()
    monkeypatch.setattr(module, "HarborReward", grader)
    result = await module.evaluate_task(
        module.CLIConfig(),
        HarborTask("task", "fix", tmp_path),
        0,
        AsyncMock(return_value=sandbox),
        tmp_path / "trial",
    )
    assert "provider failed" in result.error
    grader.assert_not_called()
    sandbox.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_pi_rejects_prepended_profile_before_model_or_sandbox_calls(tmp_path):
    task = HarborTask("task", "prepended text", tmp_path, {"harness": {"profile": "hpc-debug-v1"}})
    sandbox = AsyncMock()
    with pytest.raises(ValueError, match="original task"):
        await module.pi_rollout(module.CLIConfig(), task, sandbox, tmp_path)
    sandbox.run_command.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["hpc-debug", "swe-kokkos-debug"])
async def test_skill_is_installed_only_in_sandbox_and_upload_failure_stops(tmp_path, name):
    sandbox = AsyncMock()
    sandbox.run_command.return_value = SandboxResult(stdout="", stderr="", exit_code=0)
    sandbox.write_file.return_value = SandboxResult(stdout="", stderr="read-only", exit_code=1)
    with pytest.raises(RuntimeError, match="install HPC skill"):
        await module.pi_rollout(
            module.CLIConfig(hpc_mode="skill", hpc_skill=name),
            HarborTask("task", "original", tmp_path),
            sandbox,
            tmp_path,
        )
    path, content = sandbox.write_file.call_args.args
    assert path == f"/opt/hpc-harness/skills/{name}/SKILL.md"
    assert content.startswith(f"---\nname: {name}")


def test_harness_identity_distinguishes_arms_and_tracks_resources():
    baseline = module.harness_identity(module.CLIConfig(hpc_mode="baseline"))
    skill = module.harness_identity(module.CLIConfig(hpc_mode="skill"))
    assert baseline != skill
    assert baseline["resource_sha256"] == skill["resource_sha256"]
    assert "integration.mjs" in skill["resource_sha256"]


def test_agent_directory_explicit_overrides_environment(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "runner.mjs").write_text("")
    monkeypatch.setenv("PI_HPC_AGENT_DIR", str(tmp_path / "missing"))
    assert module.pi_agent_directory(module.CLIConfig(pi_agent_dir=str(explicit))) == explicit
    with pytest.raises(ValueError, match="set pi_agent_dir"):
        module.pi_agent_directory(module.CLIConfig())
    monkeypatch.setenv("PI_HPC_AGENT_DIR", str(explicit))
    assert module.pi_agent_directory(module.CLIConfig()) == explicit
