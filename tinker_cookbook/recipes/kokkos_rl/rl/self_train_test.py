"""Safety and token-alignment checks for the self-training experiment."""

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.recipes.kokkos_rl.rl.rollout_data import (
    RolloutRecorder,
    command_audit,
    eligible,
)
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import (
    RecordedDataset,
    RecordedDatasetBuilder,
    choose_donors,
    grade_patch,
    validate_instruction_quality,
)
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.sandbox import SandboxResult


@pytest.mark.asyncio
@pytest.mark.parametrize("upload_exit_code", [-1, 1])
async def test_failed_candidate_upload_stops_before_apply_and_cleans_up(
    tmp_path: Path, upload_exit_code: int
) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"base_commit": "a" * 40, "merge_commit": "b" * 40})
    )
    sandbox = AsyncMock()
    sandbox.run_command.return_value = SandboxResult(stdout="", stderr="", exit_code=0)
    sandbox.write_file.return_value = SandboxResult(
        stdout="", stderr="upload transport failure", exit_code=upload_exit_code
    )
    factory = AsyncMock(return_value=sandbox)
    task = HarborTask("upload-test", "Apply a production patch", tmp_path)

    with pytest.raises(RuntimeError, match="Candidate patch upload failed"):
        await grade_patch(task, factory, "candidate source patch")

    # Only the initial clean-room check ran; never apply a missing or stale patch.
    sandbox.run_command.assert_awaited_once()
    assert "git apply" not in sandbox.run_command.call_args.args[0]
    sandbox.write_file.assert_awaited_once_with("/tmp/candidate.patch", "candidate source patch")
    sandbox.cleanup.assert_awaited_once()


def test_known_instruction_contradiction_blocks_training(tmp_path: Path) -> None:
    task = HarborTask(
        "kokkos__kokkos-7043",
        "Resolve aliases, suppress anonymous-namespace\n prefixes that vary by compiler.",
        tmp_path,
    )
    with pytest.raises(ValueError, match="Instruction quality gate failed"):
        validate_instruction_quality([task])


def test_reviewed_instruction_is_accepted(tmp_path: Path) -> None:
    overrides_path = Path(__file__).parents[1] / "dataset/instruction_overrides.json"
    instruction = json.loads(overrides_path.read_text())["kokkos__kokkos-7043"]
    assert "preserve compiler-specific anonymous-namespace prefixes" in instruction
    validate_instruction_quality([HarborTask("kokkos__kokkos-7043", instruction, tmp_path)])


def donor(task="train", sample=0, turns=10):
    return {
        "task_name": task,
        "sample_index": sample,
        "turns": turns,
        "reward": 1,
        "sampled_tokens": 2,
        "stop_reason": "completed",
        "audit_flags": [],
        "datums": [{"input_tokens": [1, 2, 3], "target_tokens": [2, 3, 4], "weights": [0, 1, 1]}],
    }


def test_selection_has_identical_task_support_and_shortest_success():
    rows = [
        (Path(f"/{task}-{i}"), donor(task, i, turns))
        for task in ["a", "b"]
        for i, turns in enumerate([30, 12, 25])
    ]
    arms = choose_donors(rows)
    assert len(arms["random_success"]) == len(arms["short_success"]) == 2
    assert arms["short_success"] == ["/a-1", "/b-1"]
    assert arms == choose_donors(list(reversed(rows)))


@pytest.mark.parametrize(
    "changes",
    [
        {"reward": 0},
        {"turns": 41},
        {"stop_reason": "max_turns"},
        {"audit_flags": ["answer_lookup"]},
        {"datums": []},
    ],
)
def test_ineligible_donors_are_rejected(changes):
    assert not eligible({**donor(), **changes})


def test_masks_shift_and_heldout_exclusion(tmp_path):
    path = tmp_path / "donor.json"
    path.write_text(json.dumps(donor()))
    dataset = RecordedDataset([str(path)], 4)
    datum = dataset.get_batch(0)[0]
    assert list(datum.loss_fn_inputs["weights"].data) == [0, 0.5, 0.5]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"paths": [str(path)], "train_tasks": ["another_task"]}))
    with pytest.raises(ValueError, match="Heldout"):
        RecordedDatasetBuilder(manifest=str(manifest))()
    bad = donor()
    bad["datums"][0]["target_tokens"] = [8, 9, 10]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="shifted"):
        dataset.get_batch(0)


def test_audit_reads_structured_calls():
    def message(command):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="bash", arguments=json.dumps({"command": command})
                    ),
                    id="call1",
                )
            ],
        }

    assert command_audit([message("git diff")]) == []
    assert command_audit([message("git show HEAD")]) == ["answer_lookup"]
    assert command_audit([message("cat /tests/test.patch")]) == ["verifier_access"]
    assert command_audit([message("cat /tests/private.cpp")]) == ["verifier_access"]
    assert command_audit([message("ls /tests")]) == ["verifier_access"]
    assert command_audit([message("cat /solution/answer.cpp")]) == ["verifier_access"]
    assert command_audit([message("cat /workspace/repo/tests/test_views.py")]) == []
    assert command_audit([message("ls tests/ && ls /workspace/repo/tests/")]) == []


@pytest.mark.asyncio
async def test_recorder_preserves_action_mask_and_captures_before_grading(tmp_path):
    taskdir = tmp_path / "task"
    taskdir.mkdir()
    (taskdir / "metadata.json").write_text(json.dumps({"base_commit": "a" * 40}))
    (taskdir / "tests").mkdir()
    (taskdir / "tests/test.sh").write_text("baseline=" + "a" * 40 + "\n")
    sandbox = AsyncMock()
    sandbox.sandbox_id = "test-sandbox"
    sandbox.run_command.return_value = SandboxResult(
        stdout=json.dumps(
            {
                "base_commit": "a" * 40,
                "head_commit": "a" * 40,
                "patch_bytes": 5,
                "patch_sha256": hashlib.sha256(b"patch").hexdigest(),
                "patch_base64": base64.b64encode(b"patch").decode(),
            }
        ),
        stderr="",
        exit_code=0,
    )

    async def grade(history):
        assert sandbox.run_command.await_count == 1
        return 1.0, {"test_passed": 1.0}

    recorder = RolloutRecorder(
        HarborTask("task", "instruction", taskdir), sandbox, tmp_path, 0, grade
    )
    await recorder([])
    trajectory = Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([1, 2]),
                ac=TokensWithLogprobs(tokens=[3, 4], maybe_logprobs=[-0.2, -0.3]),
                reward=1,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.empty(),
        stop_reason="completed",
    )
    recorder.save(trajectory)
    row = json.loads((recorder.directory / "trajectory.json").read_text())
    assert row["datums"] == [
        {"input_tokens": [1, 2, 3], "target_tokens": [2, 3, 4], "weights": [0, 1, 1]}
    ]
    assert (recorder.directory / "patch.diff").read_text() == "patch"


@pytest.mark.asyncio
async def test_recorder_leaves_untracked_build_outputs_out_of_verifier_diff(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL)

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "source.hpp").write_text("before\n")
    git("add", ".")
    git("commit", "-m", "baseline")
    baseline = git("rev-parse", "HEAD").decode().strip()
    (repo / "source.hpp").write_text("fixed\n")
    (repo / "new.hpp").write_text("new production source\n")
    build = repo / "build/unit_test"
    build.mkdir(parents=True)
    (build / "output.o").write_bytes(b"x" * 2_100_000)
    index_before = (repo / ".git/index").read_bytes()
    taskdir = tmp_path / "task"
    (taskdir / "tests").mkdir(parents=True)
    (taskdir / "tests/test.sh").write_text(f"baseline={baseline}\n")

    async def run_command(command, **kwargs):
        result = subprocess.run(command, cwd=repo, shell=True, capture_output=True, text=True)
        return SandboxResult(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode
        )

    sandbox = AsyncMock()
    sandbox.sandbox_id = "local-fixture"
    sandbox.run_command.side_effect = run_command

    async def grade(history):
        assert (repo / ".git/index").read_bytes() == index_before
        # This is the verifier's tracked-diff input; build output must remain
        # untracked so the separate build exclusion still applies.
        assert git("diff", "--name-only", baseline).splitlines() == [b"source.hpp"]
        assert b"build/unit_test/output.o" in git("ls-files", "--others", "--exclude-standard")
        return 1.0, {"test_passed": 1.0}

    recorder = RolloutRecorder(
        HarborTask("task", "instruction", taskdir), sandbox, tmp_path / "results", 0, grade
    )
    reward, _ = await recorder([])
    assert reward == 1
    assert recorder.capture_error is None
    assert "new production source" in recorder.patch and "+fixed" in recorder.patch
    assert "build/unit_test" not in recorder.patch
    assert json.loads((recorder.directory / "candidate.json").read_text())["complete"]
