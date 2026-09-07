"""Negative tests for recovery authorization, selection and at-most-once dispatch."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.kokkos_rl.rl import baseline_recovery as recovery
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import pinned_file


def result(task: str, index: int, error: str | None = None) -> TaskResult:
    return TaskResult(task, index, 0.0, {}, 40, 1.0, error)


def selection_fixture() -> tuple[
    dict[str, object], dict[str, object], dict[tuple[str, int], TaskResult]
]:
    tasks = [f"task{i:02d}" for i in range(20)]
    rows = {(task, i): result(task, i) for task in tasks for i in range(4)}
    manifest: dict[str, object] = {"heldout": tasks, "task_hashes": dict.fromkeys(tasks, "hash")}
    proposal: dict[str, object] = {
        "new_rollouts": 1,
        "slots": [
            {
                "task_name": tasks[0],
                "sample_index": 0,
                "heldout_only": True,
                "task_sha256": "hash",
                "original_reward": 0.0,
                "original_error": None,
            }
        ],
    }
    return proposal, manifest, rows


def test_selection_reuses_other_slots_without_sampling() -> None:
    proposal, manifest, rows = selection_fixture()
    assert recovery.select_slots(proposal, manifest, rows) == [("task00", 0)]
    assert len(rows) == 80


@pytest.mark.parametrize(
    "change",
    ["duplicate", "outside", "boolean_index", "budget", "original_error", "missing_original"],
)
def test_invalid_selection_is_rejected(change: str) -> None:
    proposal, manifest, rows = selection_fixture()
    slots = proposal["slots"]
    assert isinstance(slots, list)
    if change == "duplicate":
        slots.append(slots[0])
    elif change == "outside":
        slots[0]["task_name"] = "train-only"
    elif change == "boolean_index":
        slots[0]["sample_index"] = False
    elif change == "budget":
        proposal["new_rollouts"] = 2
    elif change == "original_error":
        slots[0]["original_error"] = ""
    else:
        rows.pop(("task19", 3))
    with pytest.raises(ValueError):
        recovery.select_slots(proposal, manifest, rows)


def test_changed_document_fails_before_remote_work(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    path.write_text("{}")
    sha = pinned_file(path)["sha256"]
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        recovery.checked_document(str(path), sha)


@pytest.mark.parametrize(
    "artifact",
    ["attempts/task__00.json", "rollouts/task__00/messages.json", "terminal/task__00.json"],
)
def test_incomplete_attempt_is_never_resampled(tmp_path: Path, artifact: str) -> None:
    p = tmp_path / artifact
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    with pytest.raises(ValueError, match="replayed"):
        recovery.pending_slots(tmp_path, [("task", 0)])


def completed_fixture(root: Path) -> None:
    recovery.exclusive_json(root / "identity.json", {"phase": "recovery"})
    recovery.exclusive_json(
        root / "attempts/task__00.json",
        {
            "task_name": "task",
            "sample_index": 0,
            "retry_permitted": False,
            "identity": pinned_file(root / "identity.json"),
        },
    )
    row = asdict(result("task", 0))
    (root / "results.jsonl").write_text(json.dumps(row) + "\n")
    recovery.exclusive_json(root / "terminal/task__00.json", row)


def test_complete_attempt_is_skipped_but_changed_identity_is_rejected(tmp_path: Path) -> None:
    completed_fixture(tmp_path)
    assert recovery.pending_slots(tmp_path, [("task", 0), ("task", 1)]) == [("task", 1)]
    (tmp_path / "identity.json").write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        recovery.pending_slots(tmp_path, [("task", 0)])


def test_empty_error_string_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "results.jsonl").write_text(json.dumps(asdict(result("task", 0, ""))) + "\n")
    with pytest.raises(ValueError, match="failed"):
        recovery.pending_slots(tmp_path, [("task", 0)])


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_failure_stops_new_work_and_drains_inflight(tmp_path: Path, raises: bool) -> None:
    recovery.exclusive_json(tmp_path / "identity.json", {"phase": "recovery"})
    started_second = asyncio.Event()
    observed: list[int] = []

    async def operation(pair: tuple[str, int]) -> TaskResult:
        task, index = pair
        observed.append(index)
        assert (tmp_path / "attempts" / f"{task}__{index:02d}.json").exists()
        if index == 0:
            await started_second.wait()
            if raises:
                raise RuntimeError("infrastructure failure")
            return result(task, index, "")
        started_second.set()
        await asyncio.sleep(0.01)
        observed.append(99)
        return result(task, index)

    rows = await recovery.dispatch_slots(tmp_path, [("task", i) for i in range(4)], operation, 2)
    assert observed == [0, 1, 99]
    assert rows[2:] == [None, None]
    assert (tmp_path / "terminal/task__01.json").is_file()
    assert not (tmp_path / "attempts/task__02.json").exists()


@pytest.mark.parametrize(
    "key",
    ["max_turns", "thinking_effort", "max_sampled_tokens", "max_infra_retries", "allow_network"],
)
def test_changed_authorized_budget_is_rejected(tmp_path: Path, key: str) -> None:
    cfg = recovery.eval_config(tmp_path, "model", {}, 2)
    proposal: dict[str, object] = {
        "model_name": "model",
        "maximum_added_rollout_turns": 240,
        "maximum_added_sampled_tokens": 393216,
        "protocol": {
            name: getattr(cfg, name)
            for name in [
                "max_turns",
                "max_tokens",
                "max_sampled_tokens",
                "max_trajectory_tokens",
                "max_tool_calls",
                "temperature",
                "thinking_effort",
                "command_timeout",
                "grader_timeout",
                "max_infra_retries",
                "max_concurrency",
                "sandbox_build_parallelism",
                "allow_network",
            ]
        },
    }
    recovery.validate_budget(proposal, cfg, 6)
    protocol = proposal["protocol"]
    assert isinstance(protocol, dict)
    protocol[key] = "changed"
    with pytest.raises(ValueError, match="authorized protocol"):
        recovery.validate_budget(proposal, cfg, 6)
