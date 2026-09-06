import base64
import json
import subprocess
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import (
    capture_candidate,
    export_command,
)
from tinker_cookbook.sandbox import SandboxInterface


def test_candidate_export_includes_agent_commits_and_untracked_files_without_mutating_repo(
    tmp_path,
):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL
        )

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "tracked.hpp").write_text("before\n")
    git("add", ".")
    git("commit", "-m", "baseline")
    baseline = git("rev-parse", "HEAD").decode().strip()
    (tmp_path / "tracked.hpp").write_text("agent committed change\n")
    git("commit", "-am", "agent commit")
    (tmp_path / "new header.hpp").write_text("new source\n")
    before = git("status", "--porcelain")
    record = json.loads(subprocess.check_output(export_command(baseline), cwd=tmp_path, shell=True))
    assert git("status", "--porcelain") == before
    patch = base64.b64decode(record["patch_base64"])
    assert b"agent committed change" in patch and b"new source" in patch
    assert record["base_commit"] == baseline
    assert record["head_commit"] != baseline
    git("reset", "--hard", baseline)
    (tmp_path / "new header.hpp").unlink()
    subprocess.run(["git", "apply", "--binary", "-"], cwd=tmp_path, input=patch, check=True)
    assert (tmp_path / "tracked.hpp").read_text() == "agent committed change\n"
    assert (tmp_path / "new header.hpp").read_text() == "new source\n"


@pytest.mark.asyncio
async def test_incomplete_artifact_is_explicit_and_does_not_produce_a_patch(tmp_path):
    task = make_task(tmp_path, "task")
    (task.task_dir / "tests/test.sh").write_text("baseline=" + "a" * 40 + "\n")
    sandbox = SimpleNamespace(
        sandbox_id="original-image",
        run_command=AsyncMock(
            return_value=SimpleNamespace(exit_code=0, stdout='{"truncated":', stderr="")
        ),
    )
    await capture_candidate(cast(SandboxInterface, sandbox), task=task, results_dir=tmp_path)
    metadata = json.loads((tmp_path / "candidate.json").read_text())
    assert metadata["complete"] is False
    assert "JSONDecodeError" in metadata["error"]
    assert not (tmp_path / "candidate.patch").exists()
