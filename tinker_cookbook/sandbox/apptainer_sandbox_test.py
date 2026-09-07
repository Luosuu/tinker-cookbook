"""Harbor working-directory semantics without requiring Apptainer."""

import asyncio
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
    create.assert_awaited_once_with(env / "cached.sif", 300, default_workdir="/workspace/repo")


def test_factory_rejects_relative_container_workdir(tmp_path: Path) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    (tmp_path / "task.toml").write_text('[environment]\nworkdir="repo"\n')
    with pytest.raises(ValueError, match="absolute container path"):
        asyncio.run(apptainer_sandbox_factory(env, 300))
