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


def test_identity_remains_compatible_with_legacy_package(tmp_path):
    import shutil

    source = module.pi_agent_directory(module.CLIConfig())
    for name in [
        "runner.mjs",
        "integration.mjs",
        "plugins.mjs",
        "package.json",
        "skills/hpc-debug/SKILL.md",
    ]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    legacy = module.harness_identity(module.CLIConfig(pi_agent_dir=str(tmp_path)))
    assert legacy["protocol_budget_support"] is False
    assert "protocol.mjs" not in legacy["resource_sha256"]
    (tmp_path / "protocol.mjs").write_text("// protocol revision")
    modern = module.harness_identity(module.CLIConfig(pi_agent_dir=str(tmp_path)))
    assert modern["protocol_budget_support"] is True
    assert "protocol.mjs" in modern["resource_sha256"]


def test_protocol_corrections_default_to_disabled():
    assert module.CLIConfig().max_protocol_repairs == 0


def test_file_operations_are_served_inside_the_sandbox():
    sandbox = AsyncMock()
    sandbox.read_file.return_value = SandboxResult(exit_code=0, stdout="int x;\n", stderr="")
    sandbox.write_file.return_value = SandboxResult(exit_code=0, stdout="", stderr="")
    sandbox.run_command.return_value = SandboxResult(exit_code=1, stdout="", stderr="")

    def call(event):
        return asyncio.run(module.sandbox_file_op(sandbox, event))

    assert call({"op": "read", "path": "/workspace/repo/a.hpp"}) == {
        "ok": True,
        "content": "int x;\n",
    }
    assert call({"op": "write", "path": "/workspace/repo/a.hpp", "content": "int y;\n"})["ok"]
    sandbox.write_file.assert_awaited_with("/workspace/repo/a.hpp", "int y;\n", timeout=120)
    assert call({"op": "access", "path": "/workspace/repo/a b.hpp"})["ok"] is False
    assert sandbox.run_command.await_args.args[0] == (
        "test -r '/workspace/repo/a b.hpp' && test -w '/workspace/repo/a b.hpp'"
    )
    assert call({"op": "read", "path": "relative.hpp"})["ok"] is False
    assert call({"op": "delete", "path": "/workspace/repo/a.hpp"})["ok"] is False
    sandbox.read_file.return_value = SandboxResult(exit_code=1, stdout="", stderr="No such file")
    assert call({"op": "read", "path": "/x"}) == {"ok": False, "error": "No such file"}


def test_pi_tool_result_is_byte_identical_to_the_native_loop():
    """The Pi arm must show the model exactly what the native loop shows it.

    Native renders the bash result with json.dumps and keeps the first third and last two
    thirds of 16384 characters; compiler errors live in the tail.
    """
    import random
    import shutil
    import subprocess

    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import _truncate_tool_output

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    pi_dir = module.pi_agent_directory(module.CLIConfig())
    if not (pi_dir / "observation.mjs").is_file():
        pytest.skip("installed Pi agent predates native-identical tool results")
    rng = random.Random(0)
    alphabet = 'ab \n\t"\\/\x00\x1b\x7f\u00e9\u2014\u2028\U0001f600{}'
    cases = [
        {
            "exit_code": rng.choice([0, 1, 2, 124, -1]),
            "stdout": "".join(
                rng.choice(alphabet) for _ in range(rng.choice([0, 10, 9000, 20000]))
            ),
            "stderr": "".join(rng.choice(alphabet) for _ in range(rng.choice([0, 50, 17000]))),
        }
        for _ in range(12)
    ]
    script = (
        "import { toolResultText } from " + json.dumps((pi_dir / "observation.mjs").as_uri()) + ";"
        "let s='';for await (const c of process.stdin) s+=c;"
        "process.stdout.write(JSON.stringify(JSON.parse(s).map(r => toolResultText(r, 16384))));"
    )
    rendered = json.loads(
        subprocess.run(
            [node, "--input-type=module", "-e", script],
            input=json.dumps(cases),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    for case, pi_text in zip(cases, rendered, strict=True):
        native = _truncate_tool_output(json.dumps(case), 16384)
        assert pi_text == native
