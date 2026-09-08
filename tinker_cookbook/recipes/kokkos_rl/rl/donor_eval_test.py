"""Fixed heldout budget, checkpoint provenance and at-most-once boundaries."""

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import chz
import pytest

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.kokkos_rl.rl import donor_eval as evaluation
from tinker_cookbook.recipes.kokkos_rl.rl.baseline_recovery import exclusive_json
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import pinned_file


def fixed_inputs():
    heldout = [f"heldout{i:02d}" for i in range(20)]
    train = [f"train{i:02d}" for i in range(80)]
    manifest = {
        "heldout": heldout,
        "train": train,
        "task_hashes": dict.fromkeys(heldout + train, "hash"),
    }
    plan = {
        "slots": [
            {"arm": arm, "task_name": t, "sample_index": i, "task_sha256": "hash"}
            for arm in evaluation.ARMS
            for i in range(4)
            for t in heldout
        ]
    }
    return manifest, plan


def test_fixed_160_slots():
    manifest, plan = fixed_inputs()
    slots = evaluation.expected_slots(manifest, plan)
    assert len(slots) == 160
    assert len({(s["arm"], s["task_name"], s["sample_index"]) for s in slots}) == 160


@pytest.mark.parametrize(
    "change", ["duplicate", "train_leak", "wrong_hash", "order", "extra", "split"]
)
def test_changed_plan_or_split_rejected(change):
    manifest, plan = fixed_inputs()
    if change == "duplicate":
        plan["slots"][1] = plan["slots"][0]
    elif change == "train_leak":
        plan["slots"][0]["task_name"] = "train00"
    elif change == "wrong_hash":
        plan["slots"][0]["task_sha256"] = "other"
    elif change == "order":
        plan["slots"].reverse()
    elif change == "extra":
        plan["slots"].append(plan["slots"][0])
    else:
        manifest["train"][0] = manifest["heldout"][0]
    with pytest.raises(ValueError):
        evaluation.expected_slots(manifest, plan)


def budget_plan():
    return {
        "budget": {
            "arms": 2,
            "heldout_tasks_per_arm": 20,
            "samples_per_task": 4,
            "total_rollouts": 160,
            "max_turns": 40,
            "max_tokens": 16384,
            "max_sampled_tokens": 65536,
            "max_trajectory_tokens": 114688,
            "max_tool_calls": 80,
            "temperature": 1.0,
            "thinking_effort": 0.9,
            "command_timeout": 900,
            "grader_timeout": 900,
            "max_infra_retries": 0,
            "shared_sandbox_concurrency": 4,
        }
    }


def test_exact_budget_and_checkpoint(tmp_path):
    cfg = evaluation.eval_config(tmp_path, evaluation.MODEL, {}, 4)
    cfg = chz.replace(cfg, checkpoint_url="tinker://one/sampler_weights/final")
    evaluation.validate_budget(budget_plan(), cfg)
    assert cfg.checkpoint_url == "tinker://one/sampler_weights/final"
    assert cfg.max_infra_retries == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_turns", 41),
        ("max_tokens", 20000),
        ("max_sampled_tokens", 100000),
        ("thinking_effort", 0.5),
        ("max_infra_retries", 1),
        ("allow_network", True),
        ("num_samples", 5),
        ("model_name", "other"),
        ("max_concurrency", 5),
    ],
)
def test_budget_defaults_cannot_drift(tmp_path, field, value):
    cfg = evaluation.eval_config(tmp_path, evaluation.MODEL, {}, 4)
    with pytest.raises(ValueError):
        evaluation.validate_budget(budget_plan(), chz.replace(cfg, **{field: value}))


def test_receipt_pins_exact_identity(tmp_path):
    p = tmp_path / "review.json"
    identity = {"checkpoint": "one"}
    p.write_text(
        json.dumps(
            {
                "status": "accepted_checkpoint_evaluation",
                "identity_sha256": evaluation.digest(identity),
                "maximum_rollouts": 160,
                "retry_permitted": False,
            }
        )
    )
    evaluation.review_identity(p, identity)
    with pytest.raises(ValueError):
        evaluation.review_identity(p, {"checkpoint": "two"})
    with pytest.raises(ValueError):
        evaluation.review_identity(None, identity)


@pytest.mark.parametrize(
    "state,command", [("Ss", "candidate_capture_transition/resume.py"), ("Ts", "wrong.py")]
)
def test_capacity_changed_legacy_process_blocks(monkeypatch, state, command):
    monkeypatch.setattr(
        evaluation.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=f"{state} python {command}"),
    )
    with pytest.raises(ValueError):
        evaluation.capacity_snapshot()


def test_capacity_retains_unknown_reservation_after_exit(monkeypatch):
    monkeypatch.setattr(
        evaluation.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="")
    )
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda *a, **k: "")
    snapshot = evaluation.capacity_snapshot()
    assert snapshot["policy"]["reserved_legacy_unknown_sandboxes"] == 4
    assert all(r["unknown_remote_reservation_retained"] for r in snapshot["processes"])


def test_changed_source_rejected(tmp_path):
    p = tmp_path / "source"
    p.write_text("before")
    proof = pinned_file(p)
    p.write_text("after")
    with pytest.raises(ValueError):
        evaluation.check_proofs([proof])


@pytest.mark.parametrize(
    "artifact",
    ["attempts/task__00.json", "rollouts/task__00/messages.json", "terminal/task__00.json"],
)
def test_claimed_partial_cannot_be_resampled(tmp_path, artifact):
    p = tmp_path / artifact
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    with pytest.raises(ValueError):
        evaluation.pending_slots(tmp_path, [("task", 0)])


@pytest.mark.asyncio
async def test_infra_stops_new_dispatch_and_drains_existing(tmp_path):
    exclusive_json(tmp_path / "identity.json", {"checkpoint": "fixed"})
    both = asyncio.Event()
    calls = []

    async def operation(pair):
        calls.append(pair[1])
        assert (tmp_path / "attempts" / f"task__{pair[1]:02d}.json").exists()
        if pair[1] == 0:
            await both.wait()
            return TaskResult("task", 0, 0, {}, 1, 1, "")
        both.set()
        await asyncio.sleep(0.01)
        calls.append("drained")
        return TaskResult("task", 1, 1, {}, 1, 1)

    rows = await evaluation.dispatch_slots(tmp_path, [("task", i) for i in range(8)], operation, 2)
    assert calls == [0, 1, "drained"]
    assert rows[2:] == [None] * 6


@pytest.mark.asyncio
async def test_initialization_failure_records_blocked_without_retry(tmp_path, monkeypatch):
    exclusive_json(tmp_path / "identity.json", {"checkpoint": "fixed"})
    calls = []

    async def failing(config):
        calls.append(1)
        raise RuntimeError("client unavailable")

    monkeypatch.setattr(evaluation, "run", failing)
    cfg = evaluation.Config(
        collection_dir="x", training_dir="y", plan_path="z", output_dir=str(tmp_path), dispatch=True
    )
    with pytest.raises(RuntimeError):
        await evaluation.main(cfg)
    assert calls == [1]
    assert json.loads((tmp_path / "status.json").read_text())["stage"] == "blocked"
    assert len(list((tmp_path / "controller_failures").glob("*.json"))) == 1


def test_completed_error_is_not_normal_resume(tmp_path):
    row = TaskResult("task", 0, 0, {}, 1, 1, "")
    (tmp_path / "results.jsonl").write_text(json.dumps(asdict(row)) + "\n")
    with pytest.raises(ValueError):
        evaluation.pending_slots(tmp_path, [("task", 0)])


def parse_fixture(folder):
    r = folder / "rollouts/task__00"
    (r / "sampling").mkdir(parents=True)
    (r / "trajectory.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "sample_index": 0,
                "reward": 0,
                "turns": 32,
                "stop_reason": "parse_error",
            }
        )
    )
    (r / "messages.json").write_text("[]")
    (r / "sampling/000.response.json").write_text("{}")
    return r


def test_verified_parse_error_is_model_failure(tmp_path):
    parse_fixture(tmp_path)
    evaluation.check_candidate_or_model_stop(
        tmp_path, ("task", 0), TaskResult("task", 0, 0, {}, 32, 1)
    )


@pytest.mark.parametrize(
    "change", ["success", "other_stop", "wrong_slot", "no_raw", "incomplete_candidate"]
)
def test_missing_capture_is_not_masked_as_parse_error(tmp_path, change):
    r = parse_fixture(tmp_path)
    row = json.loads((r / "trajectory.json").read_text())
    if change == "success":
        row["reward"] = 1
    elif change == "other_stop":
        row["stop_reason"] = "completed"
    elif change == "wrong_slot":
        row["task_name"] = "other"
    elif change == "no_raw":
        (r / "sampling/000.response.json").unlink()
    else:
        (r / "candidate.json").write_text('{"complete":false}')
    (r / "trajectory.json").write_text(json.dumps(row))
    with pytest.raises(ValueError):
        evaluation.check_candidate_or_model_stop(
            tmp_path, ("task", 0), TaskResult("task", 0, 0, {}, 32, 1)
        )


@pytest.mark.asyncio
async def test_lock_contender_does_not_overwrite_live_status(tmp_path, monkeypatch):
    import fcntl

    exclusive_json(tmp_path / "identity.json", {"checkpoint": "fixed"})
    exclusive_json(tmp_path / "status.json", {"stage": "evaluating", "owner": 12})

    async def failing(config):
        raise BlockingIOError("already owned")

    monkeypatch.setattr(evaluation, "run", failing)
    cfg = evaluation.Config(
        collection_dir="x", training_dir="y", plan_path="z", output_dir=str(tmp_path), dispatch=True
    )
    with (tmp_path / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            await evaluation.main(cfg)
    assert json.loads((tmp_path / "status.json").read_text()) == {
        "stage": "evaluating",
        "owner": 12,
    }


def test_no_claim_initialization_failure_still_blocks_restart(tmp_path):
    (tmp_path / "status.json").write_text('{"stage":"blocked"}')
    with pytest.raises(ValueError, match="ordinary restart"):
        evaluation.check_prior_state(tmp_path, [("task", 0)])


def test_unknown_artifact_never_silently_ignored(tmp_path):
    p = tmp_path / "random_success/rollouts/train_only__00"
    p.mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown slot"):
        evaluation.check_prior_state(tmp_path, [("task", 0)])


def test_contender_failure_does_not_block_healthy_state(tmp_path):
    (tmp_path / "status.json").write_text('{"stage":"evaluating"}')
    (tmp_path / "controller_failures").mkdir()
    (tmp_path / "controller_failures/contender.json").write_text('{"error":"lock contention"}')
    evaluation.check_prior_state(tmp_path, [("task", 0)])
