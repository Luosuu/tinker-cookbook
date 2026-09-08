import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.recipes.kokkos_rl.rl import donor_qualification as dq


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path):
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / "instruction.md").write_text("Fix source")
    (folder / "task.toml").write_text("")
    write(folder / "metadata.json", {"base_commit": "a" * 40})
    task = HarborTask("task", "Fix source", folder)
    patch = tmp_path / "candidate.patch"
    patch.write_text("production patch")
    alias = tmp_path / "patch.diff"
    alias.write_bytes(patch.read_bytes())
    candidate = write(
        tmp_path / "candidate.json",
        {
            "complete": True,
            "head_commit": "a" * 40,
            "base_commit": "a" * 40,
            "patch_bytes": len(patch.read_bytes()),
            "patch_sha256": dq.pinned_file(patch)["sha256"],
        },
    )
    trajectory = write(
        tmp_path / "trajectory.json",
        {
            "task_name": "task",
            "sample_index": 0,
            "reward": 1,
            "stop_reason": "completed",
            "audit_flags": [],
            "datums": [{}],
            "turns": 2,
        },
    )
    row = {
        "task": "task",
        "sample": 0,
        "slot": "task__00",
        "task_hash": dq.current_task_digest(task),
        "technical_pass": True,
        "errors": [],
        "original_proofs": [],
        "outage_scope_decision": "review_required",
        "candidate_proof": dq.pinned_file(candidate),
        "patch_proof": dq.pinned_file(patch),
        "alias_proof": dq.pinned_file(alias),
        "trajectory_proof": dq.pinned_file(trajectory),
    }
    return task, row


@pytest.mark.parametrize(
    "change",
    ["heldout", "sample", "technical", "task_hash", "patch", "head", "stop", "flag", "partial"],
)
def test_candidate_negative(tmp_path, change):
    task, row = fixture(tmp_path)
    dq.validate_candidate(row, task, {"task"}, set())
    train, heldout = {"task"}, set()
    if change == "heldout":
        heldout = {"task"}
    elif change == "sample":
        row["sample"] = 4
    elif change == "technical":
        row["technical_pass"] = False
    elif change == "task_hash":
        row["task_hash"] = "other"
    elif change == "patch":
        Path(row["patch_proof"]["path"]).write_text("changed")
    elif change in ("head", "partial"):
        p = Path(row["candidate_proof"]["path"])
        a = json.loads(p.read_text())
        a["head_commit" if change == "head" else "complete"] = (
            "b" * 40 if change == "head" else False
        )
        write(p, a)
        row["candidate_proof"] = dq.pinned_file(p)
    else:
        p = Path(row["trajectory_proof"]["path"])
        a = json.loads(p.read_text())
        a["stop_reason" if change == "stop" else "audit_flags"] = (
            "max_turns" if change == "stop" else ["bad"]
        )
        write(p, a)
        row["trajectory_proof"] = dq.pinned_file(p)
    with pytest.raises(ValueError):
        dq.validate_candidate(row, task, train, heldout)


def test_grade_once_reuses_terminal_and_preserves_outage(tmp_path, monkeypatch):
    task, row = fixture(tmp_path)
    grader = AsyncMock(return_value=1.0)
    monkeypatch.setattr(dq, "grade_patch", grader)

    async def go():
        a = await dq.grade_once(row, task, AsyncMock(), tmp_path / "out", "identity")
        b = await dq.grade_once(row, task, AsyncMock(), tmp_path / "out", "identity")
        assert a == b and a["outage_review"] == "review_required"

    asyncio.run(go())
    assert grader.await_count == 1


@pytest.mark.parametrize("prior", ["partial", "wrongidentity", "unclaimed", "wrongresult"])
def test_partial_and_wrong_claim_never_grade(tmp_path, monkeypatch, prior):
    task, row = fixture(tmp_path)
    grader = AsyncMock(return_value=1.0)
    monkeypatch.setattr(dq, "grade_patch", grader)
    claim = {
        "identity_sha256": "identity",
        "slot": "task__00",
        "task_hash": row["task_hash"],
        "patch_sha256": row["patch_proof"]["sha256"],
        "model_requests": 0,
        "retry": False,
    }
    folder = tmp_path / "out/task__00"
    if prior == "unclaimed":
        write(folder / "grading.json", {})
    else:
        write(
            folder / "claim.json",
            {**claim, "identity_sha256": "wrong"} if prior == "wrongidentity" else claim,
        )
        if prior == "wrongresult":
            write(
                folder / "result.json",
                {"claim": {}, "stage": "complete", "reward": 1, "error": None},
            )
    with pytest.raises(ValueError):
        asyncio.run(dq.grade_once(row, task, AsyncMock(), tmp_path / "out", "identity"))
    assert grader.await_count == 0


def test_infra_persists_then_never_retries(tmp_path, monkeypatch):
    task, row = fixture(tmp_path)
    grader = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(dq, "grade_patch", grader)
    with pytest.raises(TimeoutError):
        asyncio.run(dq.grade_once(row, task, AsyncMock(), tmp_path / "out", "identity"))
    result = json.loads((tmp_path / "out/task__00/result.json").read_text())
    assert result["stage"] == "infra_error" and result["reward"] is None
    with pytest.raises(ValueError):
        asyncio.run(dq.grade_once(row, task, AsyncMock(), tmp_path / "out", "identity"))
    assert grader.await_count == 1


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_exception_stops_dispatch_but_drains_inflight(failure):
    async def go():
        active = asyncio.Event()
        failed = asyncio.Event()
        started = []
        finished = []

        async def worker(row):
            started.append(row["slot"])
            if row["slot"] == "0":
                await active.wait()
                failed.set()
                raise failure()
            active.set()
            await failed.wait()
            await asyncio.sleep(0)
            finished.append(row["slot"])
            return {"stage": "complete", "reward": 1}

        result = await dq.dispatch([{"slot": str(i)} for i in range(6)], worker, 2, lambda _: None)
        assert started == ["0", "1"] and finished == ["1"] and len(result) == 2

    asyncio.run(go())


def test_progress_failure_blocks_completion():
    async def worker(_):
        return {"stage": "complete", "reward": 1}

    def progress(_):
        raise OSError("disk")

    result = asyncio.run(dq.dispatch([{"slot": "0"}, {"slot": "1"}], worker, 1, progress))
    assert any(r["stage"] == "progress_error" for r in result) and len(result) == 2


def test_audit_frozen_object_tamper(tmp_path):
    original = tmp_path / "original"
    original.write_text("bytes")
    root = tmp_path / "audit"
    obj = root / "objects/object"
    obj.parent.mkdir(parents=True)
    obj.write_bytes(original.read_bytes())
    audit = {
        "source_proofs": {
            str(original): {
                "sha256": dq.pinned_file(original)["sha256"],
                "bytes": 5,
                "object": "objects/object",
            }
        }
    }
    assert dq.audited_source(audit, root, original) == obj
    obj.write_text("bad")
    with pytest.raises(ValueError):
        dq.audited_source(audit, root, original)


def test_post_grade_source_change_keeps_error_and_stops(tmp_path, monkeypatch):
    task, row = fixture(tmp_path)
    monkeypatch.setattr(dq, "grade_patch", AsyncMock(return_value=1.0))

    def changed():
        raise ValueError("source changed during grading")

    with pytest.raises(ValueError):
        asyncio.run(dq.grade_once(row, task, AsyncMock(), tmp_path / "out", "identity", changed))
    result = json.loads((tmp_path / "out/task__00/result.json").read_text())
    assert result["stage"] == "infra_error" and result["reward"] is None


@pytest.mark.parametrize("case", ["native_alive", "nebius_running", "nebius_missing"])
def test_capacity_refuses_unavailable_reservation(monkeypatch, case):
    from types import SimpleNamespace

    config = dq.Config(collection_dir="c", audit_path="a", output_dir="o")
    monkeypatch.setattr(dq, "pid_alive", lambda _: case == "native_alive")
    monkeypatch.setattr(
        dq.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1 if case == "nebius_missing" else 0,
            stdout="Ss" if case == "nebius_running" else "Ts",
        ),
    )
    with pytest.raises(ValueError):
        dq.capacity(config)
