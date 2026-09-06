"""Safety and token-alignment checks for the self-training experiment."""

import json
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
)
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.sandbox import SandboxResult


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


@pytest.mark.asyncio
async def test_recorder_preserves_action_mask_and_captures_before_grading(tmp_path):
    taskdir = tmp_path / "task"
    taskdir.mkdir()
    (taskdir / "metadata.json").write_text(json.dumps({"base_commit": "a" * 40}))
    sandbox = AsyncMock()
    sandbox.run_command.return_value = SandboxResult(stdout="patch", stderr="", exit_code=0)

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
