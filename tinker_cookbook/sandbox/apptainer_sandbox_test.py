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
    create.assert_awaited_once_with(
        env / "cached.sif",
        300,
        default_workdir="/workspace/repo",
        enable_gpu=False,
        allow_network=True,
        command_env=None,
    )


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
        marker = Path(kwargs["extra_bind_mounts"][0].split(":")[0])
        workspace = Mock()
        workspace.execute_command.return_value = SandboxResult(
            stdout=marker.read_text(), stderr="", exit_code=0
        )
        return workspace

    factory = Mock(side_effect=construct)
    module = Mock(ApptainerWorkspace=factory)
    with patch.dict("sys.modules", {"openhands.workspace": module}):
        asyncio.run(ApptainerSandbox.create(sif, enable_gpu=enable_gpu))
    assert factory.call_args.kwargs["enable_gpu"] is enable_gpu
    assert factory.call_args.kwargs["disable_mount_locations"] == [
        "hostfs",
        "bind-paths",
        "cwd",
        "home",
    ]
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
    workspace.execute_command.return_value = SandboxResult(
        stdout="wrong-container", stderr="", exit_code=0
    )
    factory = Mock(return_value=workspace)
    before = set(_LEASED_PORTS)
    with patch.dict("sys.modules", {"openhands.workspace": Mock(ApptainerWorkspace=factory)}):
        with pytest.raises(RuntimeError, match="identity mismatch"):
            asyncio.run(ApptainerSandbox.create(sif))
    workspace.cleanup.assert_called_once()
    assert before == _LEASED_PORTS


def test_port_lease_rejects_reuse() -> None:
    from tinker_cookbook.sandbox.apptainer_sandbox import _lease_port, _release_port

    with (
        patch("tinker_cookbook.sandbox.apptainer_sandbox.socket.socket") as socket_factory,
        patch(
            "tinker_cookbook.sandbox.apptainer_sandbox.secrets.randbelow",
            side_effect=[12001, 12001, 12002],
        ),
    ):
        first = _lease_port()
        second = _lease_port()
    try:
        assert first == 22001 and second == 22002
        assert socket_factory.return_value.__enter__.return_value.bind.call_count == 2
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
    assert args == [
        "unshare",
        "--user",
        "--map-root-user",
        "--net",
        "--",
        "/bin/bash",
        "-c",
        command,
    ]


def test_prebind_race_retries_with_a_new_port(tmp_path: Path) -> None:
    sif = tmp_path / "test.sif"
    sif.touch()
    ports = []

    def construct(**kwargs):
        ports.append(kwargs["host_port"])
        if len(ports) == 1:
            raise RuntimeError(f"Port {ports[-1]} is not available")
        marker = Path(kwargs["extra_bind_mounts"][0].split(":")[0])
        workspace = Mock()
        workspace.execute_command.return_value = SandboxResult(
            stdout=marker.read_text(), stderr="", exit_code=0
        )
        return workspace

    with patch.dict("sys.modules", {"openhands.workspace": Mock(ApptainerWorkspace=construct)}):
        asyncio.run(ApptainerSandbox.create(sif))
    assert len(ports) == 2


def test_command_environment_is_applied_inside_offline_namespace() -> None:
    import shlex

    workspace = Mock()
    workspace.execute_command.return_value = SandboxResult(stdout="", stderr="", exit_code=0)
    sandbox = ApptainerSandbox(
        workspace, 60, allow_network=False, command_env={"OMP_NUM_THREADS": "4"}
    )
    asyncio.run(sandbox.run_command("echo $OMP_NUM_THREADS"))
    outer = shlex.split(workspace.execute_command.call_args.args[0])
    assert outer[:4] == ["unshare", "--user", "--map-root-user", "--net"]
    assert shlex.split(outer[-1]) == [
        "env",
        "--",
        "OMP_NUM_THREADS=4",
        "/bin/bash",
        "-c",
        "echo $OMP_NUM_THREADS",
    ]


def test_identity_does_not_mutate_process_environment_after_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tinker_cookbook.sandbox.apptainer_sandbox._ENV_CONFIGURED", True)
    sif = tmp_path / "test.sif"
    sif.touch()

    def construct(**kwargs):
        marker = Path(kwargs["extra_bind_mounts"][0].split(":")[0])
        workspace = Mock(host_port=kwargs["host_port"])
        workspace.execute_command.return_value = SandboxResult(
            stdout=marker.read_text(), stderr="", exit_code=0
        )
        return workspace

    async def run():
        sandbox = await ApptainerSandbox.create(sif)
        marker = sandbox._identity_file
        assert marker is not None and marker.exists()
        await sandbox.cleanup()
        assert not marker.exists()

    with patch.dict("sys.modules", {"openhands.workspace": Mock(ApptainerWorkspace=construct)}):
        with patch("os.putenv", side_effect=AssertionError("concurrent environment mutation")):
            asyncio.run(run())


@pytest.mark.parametrize("data", [b"", bytes(range(256)) * 1024], ids=["empty", "large"])
def test_chunked_upload_round_trips_large_binary(tmp_path: Path, data: bytes) -> None:
    import shlex
    import subprocess
    import sys

    destination = tmp_path / "space and 'quote.bin"
    workspace = Mock()

    def execute(command, **kwargs):
        arguments = shlex.split(command)
        assert len(command) < 32768
        completed = subprocess.run([sys.executable, *arguments[1:]], capture_output=True, text=True)
        return SandboxResult(stdout=completed.stdout, stderr=completed.stderr,
                             exit_code=completed.returncode)

    workspace.execute_command.side_effect = execute
    sandbox = ApptainerSandbox(workspace, 60)
    result = asyncio.run(sandbox.write_file(str(destination), data, executable=True))
    assert result.exit_code == 0
    assert destination.read_bytes() == data
    assert destination.stat().st_mode & 0o111


def test_cleanup_can_retry_after_failure(tmp_path: Path) -> None:
    from tinker_cookbook.sandbox.apptainer_sandbox import _LEASED_PORTS, _lease_port

    port = _lease_port()
    marker = tmp_path / "identity"
    marker.touch()
    workspace = Mock(host_port=port)
    workspace.cleanup.side_effect = [RuntimeError("temporary failure"), None]
    sandbox = ApptainerSandbox(workspace, 60, leased_port=port, identity_file=marker)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="temporary failure"):
            await sandbox.cleanup()
        assert port in _LEASED_PORTS and marker.exists()
        await sandbox.cleanup()
        await sandbox.cleanup()

    asyncio.run(run())
    assert workspace.cleanup.call_count == 2
    assert port not in _LEASED_PORTS and not marker.exists()
