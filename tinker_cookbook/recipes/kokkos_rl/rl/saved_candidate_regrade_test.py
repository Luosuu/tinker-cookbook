import asyncio
import copy
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.recipes.kokkos_rl.rl import saved_candidate_regrade as rg


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path):
    tasks = []
    for phase in ["old", "new"]:
        folder = tmp_path / phase / "task"
        (folder / "tests").mkdir(parents=True)
        (folder / "environment").mkdir()
        (folder / "instruction.md").write_text("Fix the code")
        (folder / "task.toml").write_text("")
        (folder / "tests/test.sh").write_text(phase)
        (folder / "environment/Dockerfile").write_text("FROM example")
        write(folder / "metadata.json", {"base_commit": "a" * 40})
        tasks.append(HarborTask("task", "Fix the code", folder))
    old, new = tasks
    archive = rg.ProofArchive(tmp_path / "archive")
    model = rg.MODELS[0]
    patch = tmp_path / "source/candidate.patch"
    patch.parent.mkdir()
    patch.write_text("source patch")
    candidate = {
        "complete": True,
        "stage": "before_hidden_tests",
        "sandbox_id": "original-sandbox",
        "base_commit": "a" * 40,
        "head_commit": "a" * 40,
        "patch_bytes": patch.stat().st_size,
        "patch_sha256": rg.pinned_file(patch)["sha256"],
    }
    files = {
        "candidate_proof": write(patch.with_name("candidate.json"), candidate),
        "patch_proof": patch,
        "model_identity_proof": write(
            patch.with_name("identity.json"),
            {
                "identity": {
                    "config": {
                        "model_name": model,
                        "chat_provider": "nebius",
                        "api_mode": "chat",
                        "reasoning_effort": "high",
                        "temperature": None,
                        "max_tokens": 16384,
                        "max_input_tokens": 5_000_000,
                        "grader_timeout": 900,
                        "max_turns": 40,
                        "max_sampled_tokens": 65536,
                        "max_tool_calls": 80,
                        "command_timeout": 900,
                        "allow_network": False,
                    }
                },
                "tasks": {"task": rg.current_task_digest(old)},
            },
        ),
        "marker_proof": write(patch.with_name("marker.json"), {"task": "task", "model": model}),
        "result_proof": write(
            patch.with_name("results.jsonl"),
            {"task_name": "task", "turns_used": 1, "tool_calls": 1, "reward": 1, "error": None},
        ),
        "trace_proof": write(
            patch.with_name("trace.json"), [{"function_calls": [{"name": "bash"}]}]
        ),
    }
    row = {
        "model": model,
        "task": "task",
        "split": "heldout",
        "eligible_for_future_explicit_patch_only_regrade": True,
        "blockers": [],
        "candidate": candidate,
        "original_task_hash": rg.current_task_digest(old),
        "final_task_hash": rg.current_task_digest(new),
        "source_proofs": [],
        "source_file_checks": {},
        "original_resource_policy": asdict(rg.Policy()),
        "final_resource_policy": asdict(rg.Policy()),
        "verifier_identity": rg.verifier_identity(new),
        "prior_gradings": [],
        "decision": "new_verifier",
    }
    for key, path in files.items():
        proof = rg.pinned_file(path)
        archive.add(proof)
        row[key] = proof
        row["source_proofs"].append(proof)
    for name in ["instruction.md", "environment/Dockerfile", "task.toml"]:
        a, b = rg.pinned_file(old.task_dir / name), rg.pinned_file(new.task_dir / name)
        archive.add(a)
        archive.add(b)
        row["source_file_checks"][name] = {"identical": True, "original": a, "final": b}
    return row, archive, new, old, {"train": [], "heldout": ["task"]}


def test_archive_survives_original_changes_but_rejects_frozen_tampering(tmp_path):
    p = tmp_path / "source"
    p.write_text("original")
    proof = rg.pinned_file(p)
    archive = rg.ProofArchive(tmp_path / "frozen")
    archive.add(proof)
    p.write_text("later mutable source")
    assert archive.resolve(proof).read_text() == "original"
    with pytest.raises(ValueError, match="before freezing"):
        archive.add(proof)
    archive.resolve(proof).write_text("tampered copy")
    with pytest.raises(ValueError, match="differs"):
        archive.verify()


def test_archive_does_not_accept_external_copy_path(tmp_path):
    p = tmp_path / "external"
    p.write_text("data")
    proof = rg.pinned_file(p)
    archive = rg.ProofArchive(tmp_path / "frozen", {str(p): proof})
    with pytest.raises(ValueError, match="escapes"):
        archive.resolve(proof)


def test_candidate_positive_review_and_unchanged_model_inputs(tmp_path):
    row, archive, task, old, split = fixture(tmp_path)
    assert rg.verify_record(row, archive, task, old, split).read_text() == "source patch"
    (task.task_dir / "instruction.md").write_text("a different problem")
    with pytest.raises(ValueError):
        rg.verify_record(row, archive, task, old, split)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "train"),
        ("eligible_for_future_explicit_patch_only_regrade", False),
        ("final_resource_policy", {"backend": "modal"}),
        ("final_task_hash", "wrong"),
        ("verifier_identity", {}),
        ("decision", "reuse_existing_score"),
    ],
)
def test_changed_candidate_contract_fails_closed(tmp_path, field, value):
    row, archive, task, old, split = fixture(tmp_path)
    row[field] = value
    with pytest.raises(ValueError):
        rg.verify_record(row, archive, task, old, split)


def test_prior_wrong_answer_is_reused_not_retried_and_changed_verifier_is_distinct():
    prior = {
        "stage": "complete",
        "error": None,
        "reward": 0.0,
        "patch_sha256": "patch",
        "verifier_identity": {"tests": "old"},
    }
    assert rg.prior_disposition(prior, {"tests": "old"}, "patch") == "reuse_existing_score"
    assert rg.prior_disposition(prior, {"tests": "new"}, "patch") == "new_verifier"
    with pytest.raises(ValueError):
        rg.prior_disposition(prior, {"tests": "new"}, "different patch")
    prior["error"] = ""
    assert rg.prior_disposition(prior, {"tests": "old"}, "patch") == "blocked_prior_attempt"


def test_reused_pass_keeps_reward_and_exact_source_and_rejects_conflicting_scores():
    prior = {
        "stage": "complete",
        "error": None,
        "reward": 1.0,
        "patch_sha256": "patch",
        "verifier_identity": {"tests": "same"},
        "status_proof": {"path": "evidence", "sha256": "hash"},
    }
    row = {
        "model": "m",
        "task": "t",
        "split": "heldout",
        "candidate": {"patch_sha256": "patch"},
        "verifier_identity": {"tests": "same"},
        "prior_gradings": [prior],
    }
    result = rg.reused_result(row)
    assert (
        result["reward"] == 1
        and result["prior_gradings"][0]["status_proof"] == prior["status_proof"]
    )
    other = copy.deepcopy(prior)
    other["reward"] = 0
    row["prior_gradings"].append(other)
    with pytest.raises(ValueError, match="contradictory"):
        rg.reused_result(row)


@pytest.mark.parametrize("reward,error", [(0, None), (1, None), (None, TimeoutError())])
def test_once_marker_precedes_grading_and_no_terminal_result_is_retried(
    tmp_path, monkeypatch, reward, error
):
    row, archive, task, _, _ = fixture(tmp_path)
    patch = archive.resolve(row["patch_proof"])
    folder = tmp_path / "output"
    calls = []

    async def grade(*args):
        assert (folder / "attempt_started.json").exists()
        calls.append(True)
        if error is not None:
            raise error
        return reward

    monkeypatch.setattr(rg, "grade_patch", grade)
    first = asyncio.run(rg.grade_once(row, task, AsyncMock(), patch, folder, "identity"))
    second = asyncio.run(rg.grade_once(row, task, AsyncMock(), patch, folder, "identity"))
    assert first == second and len(calls) == 1
    assert first["model_requests"] == 0 and first["added_inference_cost_usd"] == 0
    if error is not None:
        assert first["stage"] == "infra_error" and first["error"] == "TimeoutError: "


def test_interrupted_attempt_is_not_resumed(tmp_path, monkeypatch):
    row, archive, task, _, _ = fixture(tmp_path)
    folder = tmp_path / "output"
    rg.exclusive_json(folder / "attempt_started.json", {})
    grade = AsyncMock()
    monkeypatch.setattr(rg, "grade_patch", grade)
    with pytest.raises(ValueError, match="Interrupted"):
        asyncio.run(
            rg.grade_once(
                row, task, AsyncMock(), archive.resolve(row["patch_proof"]), folder, "identity"
            )
        )
    grade.assert_not_called()


def test_guard_failure_drains_active_work_and_stops_new_dispatch():
    async def scenario():
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        finished = []

        async def worker(row):
            if row["task"] == "active":
                active_started.set()
                await release_active.wait()
                finished.append("active")
                return {"stage": "complete", "reward": 1}
            if row["task"] == "guard":
                await active_started.wait()
                release_active.set()
                raise ValueError("capacity changed")
            raise AssertionError("Queued task must not start")

        rows = [{"model": "m", "task": name} for name in ["guard", "active", "queued"]]
        results = await rg.drain_dispatch(rows, worker, 2, lambda _: None)
        assert finished == ["active"]
        assert {r["stage"] for r in results} == {
            "guard_or_partial_failure",
            "complete",
            "blocked_not_started",
        }

    asyncio.run(scenario())


def test_code_identity_pins_scheduler_source():
    assert rg.code_identity()["runner_sha256"] == rg.pinned_file(Path(rg.__file__))["sha256"]
