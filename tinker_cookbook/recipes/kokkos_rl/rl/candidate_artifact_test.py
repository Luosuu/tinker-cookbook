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
    verifier_baseline,
)
from tinker_cookbook.sandbox import SandboxInterface

BASE = "0410363d77b78d32937a8b77e01c8373cf90230d"
OTHER = "708abe21a7e9759cf52560d076e48b4b4562ec29"


def _v24_verifier(*, embedded: str = BASE, declared: str = BASE) -> str:
    """The v2.4 shape: the baseline line lives inside an escaped script string."""
    check = json.dumps({"command": f"cd /workspace/repo || exit 1\nbaseline={embedded}\n"})
    plan = json.dumps({"task": "kokkos__kokkos-6375", "base_commit": declared, "checks": [check]})
    return f"#!/usr/bin/env bash\npython3 - <<'S'\nrun_suite(json.loads({plan!r}))\nS\n"


def test_baseline_from_a_plain_export_line():
    assert verifier_baseline(f"set -e\nbaseline={BASE}\ngit diff $baseline\n") == BASE


def test_baseline_from_the_v24_embedded_plan():
    assert verifier_baseline(_v24_verifier()) == BASE


@pytest.mark.parametrize(
    "script",
    [
        _v24_verifier(embedded=OTHER),
        _v24_verifier(declared=OTHER) + f'"base_commit": "{BASE}"',
        f"baseline={BASE}\nbaseline={OTHER}\n",
        "no baseline here",
    ],
)
def test_ambiguous_or_missing_baseline_is_refused(script):
    with pytest.raises(ValueError, match="one immutable baseline"):
        verifier_baseline(script)


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
