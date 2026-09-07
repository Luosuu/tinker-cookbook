import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl import validation_recovery as recovery
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot_test import task_with_metadata


def pinned(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def proposal(tmp_path, kind="cancelled"):
    task = task_with_metadata(tmp_path / "tasks", "task")
    source = tmp_path / "original"
    folder = source / task.task_name
    op_id = "01a079b2-fb06-7086-832a-9559a7dbce07"
    oracle = kind == "cancelled"
    stage = "oracle" if oracle else "nop"
    original = {
        "task": task.task_name,
        "task_hash": recovery._task_digest(task),
        **asdict(recovery.Policy()),
        "stage": "error",
        "failed_stage": stage,
        "passed": False,
        "nop": 0 if oracle else None,
    }
    record = pinned(source / "validated/task.json", original)
    if kind == "staging":
        evidence = [
            record,
            pinned(
                folder / "nop.json",
                {"exit_code": 127, "stderr": "/tests/test.sh: No such file or directory"},
            ),
        ]
        reason = "verifier_never_executed_due_to_failed_staging"
    else:
        failed = pinned(
            folder / f"{stage}.json",
            {"exit_code": -1, "stderr": f"ApiTimeoutError /operations/{op_id}"},
        )
        operation = {
            "operation_id": op_id,
            "source_sha256": failed["sha256"],
            "status": "CANCELLED" if oracle else "SUCCESS",
            "result_image": None if oracle else "immutable-image",
            "reward_text": "0",
            "state": {"exit_code": 0, "timed_out": False},
        }
        evidence = [record, pinned(tmp_path / "readback/operation.json", operation)]
        if oracle:
            evidence.append(pinned(folder / "nop.json", {"exit_code": 0}))
            reason = "original_oracle_cancelled_by_sdk_after_status_transport_error"
        else:
            evidence.append(pinned(tmp_path / "readback/reward.txt", "0\n"))
            reason = "original_nop_completed_recovered_readonly_oracle_never_submitted"
    item = {
        "task": task.task_name,
        "task_hash": recovery._task_digest(task),
        "resource_policy": asdict(recovery.Policy()),
        "proposed_model_requests": 0,
        "maximum_new_stage_attempts": 1,
        "original_failed_stage": stage,
        "reason": reason,
        "original_operation_id": op_id,
        "evidence": evidence,
        "stages_to_execute": ["nop", "oracle"] if kind == "staging" else ["oracle"],
        "reuse_nop": None if kind == "staging" else 0,
    }
    return task, item


@pytest.mark.parametrize("kind", ["cancelled", "completed", "staging"])
def test_verified_recovery_stages_match_original_missing_work(tmp_path, kind):
    task, item = proposal(tmp_path, kind)
    plan = recovery.verify_plan(task, item)
    assert list(plan.stages) == item["stages_to_execute"]
    assert recovery.plan_record(plan) == json.loads(json.dumps(recovery.plan_record(plan)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_hash", "changed"),
        ("maximum_new_stage_attempts", 2),
        ("proposed_model_requests", 1),
        ("stages_to_execute", ["nop", "oracle"]),
        ("resource_policy", {"backend": "modal"}),
    ],
)
def test_identity_or_scope_expansion_rejected(tmp_path, field, value):
    task, item = proposal(tmp_path)
    item[field] = value
    with pytest.raises(ValueError):
        recovery.verify_plan(task, item)


def test_changed_source_evidence_rejected(tmp_path):
    task, item = proposal(tmp_path)
    Path(item["evidence"][0]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        recovery.verify_plan(task, item)


@pytest.mark.parametrize("kind", ["cancelled", "completed"])
def test_operation_proof_must_match_requested_stage(tmp_path, kind):
    task, item = proposal(tmp_path, kind)
    path = Path(item["evidence"][1]["path"])
    operation = json.loads(path.read_text())
    operation["status"] = "SUCCESS" if kind == "cancelled" else "CANCELLED"
    item["evidence"][1] = pinned(path, operation)
    with pytest.raises(ValueError):
        recovery.verify_plan(task, item)


def test_completed_nop_does_not_authorize_repeating_an_oracle(tmp_path):
    task, item = proposal(tmp_path, "completed")
    pinned(tmp_path / "original/task/oracle.json", {"exit_code": 0})
    with pytest.raises(ValueError, match="unsubmitted Oracle"):
        recovery.verify_plan(task, item)


@pytest.mark.asyncio
async def test_oracle_only_and_completed_resume_never_repeat_sampling_or_grading(
    tmp_path, monkeypatch
):
    task, item = proposal(tmp_path)
    plan = recovery.verify_plan(task, item)
    grader = AsyncMock(return_value=1.0)
    monkeypatch.setattr(recovery, "grade_patch", grader)
    folder = tmp_path / "out"
    result = await recovery.recover_one(task, AsyncMock(), folder, plan, "commit1")
    assert result["passed"] is True and result["nop"] == 0
    assert grader.await_args.args[2] == "gold patch"
    assert len(result["stage_evidence"]["oracle"]) == 2
    again = await recovery.recover_one(task, AsyncMock(), folder, plan, "commit2")
    assert again["passed"] is True
    assert grader.await_count == 1


@pytest.mark.asyncio
async def test_partial_marker_blocks_without_remote_calls(tmp_path, monkeypatch):
    task, item = proposal(tmp_path)
    plan = recovery.verify_plan(task, item)
    folder = tmp_path / "out"
    pinned(folder / "stages/oracle/attempt_started.json", {})
    grader = AsyncMock()
    monkeypatch.setattr(recovery, "grade_patch", grader)
    result = await recovery.recover_one(task, AsyncMock(), folder, plan, "commit")
    assert result["stage"] == "interrupted" and not result["passed"]
    grader.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [1.0, RuntimeError("infrastructure")])
async def test_bad_nop_stops_oracle_and_is_never_retried(tmp_path, monkeypatch, outcome):
    task, item = proposal(tmp_path, "staging")
    plan = recovery.verify_plan(task, item)
    grader = (
        AsyncMock(side_effect=outcome)
        if isinstance(outcome, Exception)
        else AsyncMock(return_value=outcome)
    )
    monkeypatch.setattr(recovery, "grade_patch", grader)
    for _ in range(2):
        result = await recovery.recover_one(task, AsyncMock(), tmp_path / "out", plan, "commit")
        assert not result["passed"] and result["failed_stage"] == "nop"
    assert grader.await_count == 1


@pytest.mark.asyncio
async def test_mutated_evidence_after_planning_blocks_dispatch(tmp_path, monkeypatch):
    task, item = proposal(tmp_path)
    plan = recovery.verify_plan(task, item)
    Path(item["evidence"][2]["path"]).write_text("changed")
    grader = AsyncMock()
    monkeypatch.setattr(recovery, "grade_patch", grader)
    with pytest.raises(ValueError, match="evidence changed"):
        await recovery.recover_one(task, AsyncMock(), tmp_path / "out", plan, "commit")
    grader.assert_not_awaited()
