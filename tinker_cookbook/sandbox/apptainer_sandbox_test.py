"""Harbor working-directory semantics without requiring Apptainer."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tinker_cookbook.sandbox.apptainer_sandbox import (
    ApptainerSandbox,
    apptainer_sandbox_factory,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult


def test_default_workdir_and_explicit_verifier_override() -> None:
    workspace = Mock()
    workspace.execute_command.return_value = SandboxResult(stdout="", stderr="", exit_code=0)
    sandbox = ApptainerSandbox(workspace, 60, default_workdir="/workspace/repo")

    async def run() -> None:
        await sandbox.run_command("git status")
        await sandbox.run_command("bash /tests/test.sh", workdir="/root")

    asyncio.run(run())
    assert [call.kwargs["cwd"] for call in workspace.execute_command.call_args_list] == [
        "/workspace/repo",
        "/root",
    ]


def test_factory_reads_task_workdir_and_relative_sif(tmp_path: Path) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    (tmp_path / "task.toml").write_text('[environment]\nworkdir="/workspace/repo"\n')
    (env / "sif.path").write_text("cached.sif")
    with patch.object(ApptainerSandbox, "create", new_callable=AsyncMock) as create:
        asyncio.run(apptainer_sandbox_factory(env, 300))
    create.assert_awaited_once_with(env / "cached.sif", 300, default_workdir="/workspace/repo", enable_gpu=False, allow_network=True)


def test_factory_rejects_relative_container_workdir(tmp_path: Path) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    (tmp_path / "task.toml").write_text('[environment]\nworkdir="repo"\n')
    with pytest.raises(ValueError, match="absolute container path"):
        asyncio.run(apptainer_sandbox_factory(env, 300))


@pytest.mark.parametrize("gpus", [0, 1, 2])
def test_factory_gpu_passthrough_from_task(tmp_path: Path, gpus: int) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    (tmp_path / "task.toml").write_text(f"[environment]\ngpus={gpus}\n")
    with patch.object(ApptainerSandbox, "create", new_callable=AsyncMock) as create:
        asyncio.run(apptainer_sandbox_factory(env, 300))
    assert create.await_args.kwargs["enable_gpu"] is (gpus > 0)


@pytest.mark.parametrize("value", ["-1", "true", '"1"'])
def test_factory_rejects_invalid_gpu_count(tmp_path: Path, value: str) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    (tmp_path / "task.toml").write_text(f"[environment]\ngpus={value}\n")
    with pytest.raises(ValueError, match="nonnegative integer"):
        asyncio.run(apptainer_sandbox_factory(env, 300))


@pytest.mark.parametrize("enable_gpu", [False, True])
def test_create_preserves_slurm_device_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enable_gpu: bool
) -> None:
    sif = tmp_path / "test.sif"
    sif.touch()
    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-test-device")
    def construct(**kwargs):
        marker = next(k for k in kwargs["forward_env"] if k.startswith("OH_SANDBOX_"))
        workspace = Mock()
        workspace.execute_command.return_value = SandboxResult(stdout=os.environ[marker], stderr="", exit_code=0)
        return workspace
    factory = Mock(side_effect=construct)
    module = Mock(ApptainerWorkspace=factory)
    with patch.dict("sys.modules", {"openhands.workspace": module}):
        asyncio.run(ApptainerSandbox.create(sif, enable_gpu=enable_gpu))
    assert factory.call_args.kwargs["enable_gpu"] is enable_gpu
    assert ("CUDA_VISIBLE_DEVICES" in factory.call_args.kwargs["forward_env"]) is enable_gpu
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "MIG-test-device"


def test_gpu_creation_rejects_slurm_job_without_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    factory = Mock()
    with patch.dict("sys.modules", {"openhands.workspace": Mock(ApptainerWorkspace=factory)}):
        with pytest.raises(RuntimeError, match="request GPUs first"):
            asyncio.run(ApptainerSandbox.create(tmp_path / "unused.sif", enable_gpu=True))
    factory.assert_not_called()


def test_endpoint_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    from tinker_cookbook.sandbox.apptainer_sandbox import _LEASED_PORTS
    sif = tmp_path / "test.sif"
    sif.touch()
    workspace = Mock()
    workspace.execute_command.return_value = SandboxResult(stdout="wrong-container", stderr="", exit_code=0)
    factory = Mock(return_value=workspace)
    before = set(_LEASED_PORTS)
    with patch.dict("sys.modules", {"openhands.workspace": Mock(ApptainerWorkspace=factory)}):
        with pytest.raises(RuntimeError, match="identity mismatch"):
            asyncio.run(ApptainerSandbox.create(sif))
    workspace.cleanup.assert_called_once()
    assert _LEASED_PORTS == before


def test_port_lease_rejects_reuse() -> None:
    from tinker_cookbook.sandbox.apptainer_sandbox import _lease_port, _release_port
    with patch("tinker_cookbook.sandbox.apptainer_sandbox.socket.socket"), patch("tinker_cookbook.sandbox.apptainer_sandbox.secrets.randbelow", side_effect=[12001,12001,12002]):
        first = _lease_port()
        second = _lease_port()
    try:
        assert first == 22001 and second == 22002
    finally:
        _release_port(first)
        _release_port(second)


def test_offline_command_is_quoted_in_network_namespace() -> None:
    import shlex
    workspace = Mock()
    workspace.execute_command.return_value = SandboxResult(stdout="", stderr="", exit_code=0)
    sandbox = ApptainerSandbox(workspace, 60, allow_network=False)
    command = "echo 'hello'; echo $(pwd)"
    asyncio.run(sandbox.run_command(command))
    args = shlex.split(workspace.execute_command.call_args.args[0])
    assert args == ["unshare", "--user", "--map-root-user", "--net", "--", "/bin/bash", "-c", command]
