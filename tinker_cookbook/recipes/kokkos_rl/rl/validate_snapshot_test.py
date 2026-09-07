import asyncio
import json
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
from tinker_cookbook.recipes.kokkos_rl.rl import validate_snapshot as gate


def task_with_metadata(root, name, metadata=None):
    task = make_task(root, name)
    (task.task_dir / "metadata.json").write_text(json.dumps({"metadata": metadata or {}}))
    (task.task_dir / "solution").mkdir()
    (task.task_dir / "solution/gold.patch").write_text("gold patch")
    return task


def test_gpu_requirements_and_pinned_resources(tmp_path):
    task = task_with_metadata(tmp_path, "kokkos__kokkos-8989", {"requires_gpu": True})
    assert gate.policy_for_task(task) == gate.Policy("modal", 4, 900, 16384, 4.0, "L4")
    other = task_with_metadata(tmp_path, "unreviewed-gpu-task", {"requires_gpu": True})
    with pytest.raises(ValueError, match="no reviewed GPU"):
        gate.policy_for_task(other)


def test_large_allocation_control_requires_larger_vm(tmp_path):
    task = task_with_metadata(tmp_path, "kokkos__kokkos-7074")
    policy = gate.policy_for_task(task)
    assert policy == gate.Policy("modal", 4, 900, 16384, 4.0)
    old_failure = {
        "task": task.task_name,
        "task_hash": "same-payload",
        **asdict(gate.Policy()),
        "nop": 0,
        "oracle": 0,
        "passed": False,
    }
    assert not gate.matching_evidence(old_failure, task.task_name, "same-payload", policy)


@pytest.mark.parametrize(
    "field,wrong",
    [
        ("task_hash", "other"),
        ("task", "other"),
        ("gpu", None),
        ("memory_mb", 4096),
        ("cpu", 1),
        ("build_parallelism", 1),
        ("grader_timeout", 1800),
    ],
)
def test_prior_evidence_requires_same_payload_and_resources(field, wrong):
    policy = gate.Policy("modal", 4, 900, 16384, 4.0, "L4")
    record = {
        "task": "task",
        "task_hash": "hash",
        **asdict(policy),
        "nop": 0,
        "oracle": 1,
        "passed": True,
    }
    assert gate.matching_evidence(record, "task", "hash", policy)
    assert not gate.matching_evidence({**record, field: wrong}, "task", "hash", policy)
    assert not gate.passed_evidence({**record, "nop": 1})


@pytest.mark.asyncio
async def test_partial_marker_never_restarts_a_verifier(tmp_path, monkeypatch):
    task = task_with_metadata(tmp_path / "tasks", "task")
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "attempt_started.json").write_text("{}")
    grader = AsyncMock()
    monkeypatch.setattr(gate, "grade_patch", grader)
    with pytest.raises(RuntimeError, match="never silently retry"):
        await gate.validate_one(
            task, AsyncMock(), folder, gate.Policy(), gate._task_digest(task), "commit"
        )
    grader.assert_not_awaited()


class FakeFactory:
    def __init__(self, *args, **kwargs):
        pass

    def import_cache(self, source):
        pass

    async def __call__(self, *args, **kwargs):
        raise AssertionError("No remote sandbox should be created by offline tests")


@pytest.mark.asyncio
async def test_bulk_reuses_passed_blocks_failed_and_limits_concurrency(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    tasks = [task_with_metadata(snapshot / "tasks", str(i)) for i in range(7)]
    hashes = {task.task_name: gate._task_digest(task) for task in tasks}
    (snapshot / "manifest.json").write_text(json.dumps({"task_hashes": hashes}))
    prior = tmp_path / "prior"
    prior.mkdir()
    for number, passed in ((0, True), (1, False)):
        (prior / f"{number}.json").write_text(
            json.dumps(
                {
                    "task": str(number),
                    "task_hash": hashes[str(number)],
                    **asdict(gate.Policy()),
                    "passed": passed,
                    "nop": 0,
                    "oracle": int(passed),
                }
            )
        )
    monkeypatch.setattr(gate, "load_harbor_tasks_from_dir", lambda _: tasks)
    monkeypatch.setattr(gate, "ImportedCacheFactory", FakeFactory)
    monkeypatch.setattr(gate, "load_env_file", lambda _: None)
    active = maximum = calls = 0

    async def grade(task, factory, patch, path):
        nonlocal active, maximum, calls
        assert task.task_name not in {"0", "1"}
        calls += 1
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return float(patch is not None)

    monkeypatch.setattr(gate, "grade_patch", grade)
    config = gate.Config(
        snapshot_dir=str(snapshot),
        output_path=str(tmp_path / "out"),
        reuse_dirs=(str(prior),),
        max_concurrency=2,
    )
    await gate.main(config)
    assert maximum == 2
    assert calls == 10
    results = json.loads((tmp_path / "out/result.json").read_text())
    assert results["0"]["stage"] == "reused"
    assert results["1"]["stage"] == "blocked_prior_validation"
    await gate.main(config)
    assert calls == 10
    assert len((tmp_path / "out/launch_history.jsonl").read_text().splitlines()) == 2


def test_cuda_runtime_policy_is_explicit_and_old_cpu_evidence_cannot_qualify(tmp_path):
    task = task_with_metadata(tmp_path, "kokkos__kokkos-9159")
    old_policy = gate.policy_for_task(task)
    runtime_policy = gate.policy_for_task(task, "runtime_v10")
    assert old_policy == gate.Policy()
    assert runtime_policy == gate.Policy("modal", 4, 900, 16384, 4.0, "L4")
    old_record = {
        "task": task.task_name,
        "task_hash": "same-payload",
        **asdict(old_policy),
        "nop": 0,
        "oracle": 1,
        "passed": True,
    }
    assert not gate.matching_evidence(old_record, task.task_name, "same-payload", runtime_policy)
    with pytest.raises(ValueError, match="Unknown reviewed"):
        gate.policy_for_task(task, "unreviewed")
