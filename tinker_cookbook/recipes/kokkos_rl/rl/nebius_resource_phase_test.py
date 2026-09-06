import fcntl
import json
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
from tinker_cookbook.recipes.kokkos_rl.rl import nebius_resource_phase as phase


@pytest.mark.parametrize(
    "field,wrong",
    [
        ("task_hash", "other"),
        ("backend", "contree"),
        ("memory_mb", 4096),
        ("cpu", 2),
        ("build_parallelism", 1),
        ("grader_timeout", 1800),
        ("oracle", 0),
        ("nop", 1),
        ("passed", False),
    ],
)
def test_resource_gate_rejects_evidence_for_different_execution_policy(field, wrong):
    record = {
        "task": "task",
        "task_hash": "hash",
        "passed": True,
        "nop": 0.0,
        "oracle": 1.0,
        "backend": "modal",
        "memory_mb": 16384,
        "cpu": 4.0,
        "build_parallelism": 4,
        "grader_timeout": 900,
    }
    phase.validate_resource_record(record, task_name="task", task_hash="hash")
    with pytest.raises(ValueError, match="exact task"):
        phase.validate_resource_record({**record, field: wrong}, task_name="task", task_hash="hash")


@pytest.mark.asyncio
async def test_resource_phase_rejects_start_while_original_sweep_runs(tmp_path, monkeypatch):
    run = AsyncMock()
    monkeypatch.setattr(phase, "run_phase", run)
    config = phase.ResourcePhaseConfig(
        source_root=str(tmp_path), task_names=("task",), validation_files=()
    )
    with (tmp_path / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="shared inference slots"):
            await phase.main(config)
        run.assert_not_awaited()
    await phase.main(config)
    run.assert_awaited_once_with(config)


@pytest.mark.asyncio
async def test_original_partial_attempt_blocks_resource_resampling_before_any_api(
    tmp_path, monkeypatch
):
    tasks_dir = tmp_path / "tasks"
    task = make_task(tasks_dir, "task")
    monkeypatch.setattr(phase, "load_harbor_tasks_from_dir", lambda _: [task])
    hashes = {task.task_name: phase._task_digest(task)}
    (tmp_path / "launch.json").write_text(
        json.dumps(
            {"config": {"tasks_dir": str(tasks_dir)}, "source_manifest": {"task_hashes": hashes}}
        )
    )
    source = tmp_path / phase.MODELS[0].split("/")[-1]
    source.mkdir()
    (source / "eval_identity.json").write_text(
        json.dumps(
            {
                "tasks": hashes,
                "identity": {
                    "config": {
                        "model_name": phase.MODELS[0],
                        "chat_provider": "nebius",
                        "grader_timeout": 900,
                    }
                },
            }
        )
    )
    (source / "task").mkdir()
    (source / "task/attempt_started.json").write_text("{}")
    evidence = tmp_path / "verified.json"
    evidence.write_text(
        json.dumps(
            {
                "task": "task",
                "task_hash": hashes["task"],
                "passed": True,
                "nop": 0.0,
                "oracle": 1.0,
                "backend": "modal",
                "memory_mb": 16384,
                "cpu": 4.0,
                "build_parallelism": 4,
                "grader_timeout": 900,
            }
        )
    )
    client = AsyncMock()
    monkeypatch.setattr(phase, "AsyncOpenAI", client)
    with pytest.raises(ValueError, match="resample an original attempt"):
        await phase.main(
            phase.ResourcePhaseConfig(
                source_root=str(tmp_path), task_names=("task",), validation_files=(str(evidence),)
            )
        )
    client.assert_not_called()
