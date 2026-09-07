import json
from dataclasses import asdict

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl import nebius_phase_ledger as ledger
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot_test import task_with_metadata

MODEL = ledger.MODELS[0]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path):
    task = task_with_metadata(tmp_path / "snapshot/tasks", "kokkos__kokkos-6375")
    (task.task_dir / "instruction.md").write_text(task.instruction)
    (task.task_dir / "task.toml").write_text("")
    digest = ledger._task_digest(task)
    old = tmp_path / "original"
    manifest = write(
        old / "manifest.json", {"task_hashes": {task.task_name: digest, "reserved": "hash"}}
    )
    write(
        old / "launch.json",
        {
            "config": {
                "source_manifest": str(manifest),
                "modal_task_names": [],
                "validation_dir": str(old / "validation"),
            }
        },
    )
    write(old / "validation/reserved.json", {"passed": True})
    source = write(
        tmp_path / "source.json",
        {
            "task": task.task_name,
            "task_hash": digest,
            **asdict(ledger.Policy()),
            "nop": 0,
            "oracle": 1,
            "passed": True,
            "runtime_environment_policy_version": "dockerfile_env_v1",
        },
    )
    hashes = {task.task_name: digest}
    q = write(
        tmp_path / "qualification.json",
        {
            "task_hashes": hashes,
            "tasks": {
                task.task_name: {
                    "qualified": True,
                    "successful_evidence": [ledger.pinned_file(source)],
                }
            },
        },
    )
    c = write(
        tmp_path / "coverage.json", {"task_hashes": hashes, "status": "pending", "blockers": []}
    )
    review = write(tmp_path / "manual_review.json", {"decision": "reviewed configured scope"})
    allowlist = write(
        tmp_path / "scope.json",
        {
            "approvals": {
                task.task_name: {
                    "task": task.task_name,
                    "task_hash": digest,
                    "status": "accepted",
                    "sources": [ledger.pinned_file(review)],
                }
            }
        },
    )
    kwargs = {
        "task": task,
        "model": MODEL,
        "original_root": old,
        "reserved_original_tasks": frozenset({"reserved"}),
        "qualification_path": q,
        "coverage_path": c,
        "scope_approval_path": allowlist,
        "policy": ledger.Policy(),
        "required_environment_policy": "dockerfile_env_v1",
    }
    return task, source, kwargs


def test_eligible_unattempted_pair_and_atomic_cross_phase_claim(tmp_path):
    task, _, kwargs = fixture(tmp_path)
    evidence = ledger.eligible_evidence(**kwargs)
    claim = {
        "task": task,
        "ledger_root": tmp_path / "claims",
        "phase_root": tmp_path / "phase1",
        "original_root": kwargs["original_root"],
        "reserved_original_tasks": kwargs["reserved_original_tasks"],
        "evidence": evidence,
        "phase_identity_sha256": "identity",
    }
    path = ledger.claim_pair(**claim)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        ledger.claim_pair(**{**claim, "phase_root": tmp_path / "phase2"})
    assert path.read_bytes() == before


def test_reserving_whole_old_queue_blocks_unattempted_tasks(tmp_path):
    task, _, kwargs = fixture(tmp_path)
    kwargs["reserved_original_tasks"] |= {task.task_name}
    with pytest.raises(ValueError, match="even without an attempt"):
        ledger.eligible_evidence(**kwargs)


def test_original_gate_may_shrink_but_not_expand(tmp_path):
    task, _, kwargs = fixture(tmp_path)
    old = kwargs["original_root"]
    (old / "validation/reserved.json").unlink()
    ledger.eligible_evidence(**kwargs)
    write(old / "validation" / f"{task.task_name}.json", {"passed": True})
    with pytest.raises(ledger.OriginalGateExpandedError):
        ledger.eligible_evidence(**kwargs)


@pytest.mark.parametrize(
    "filename",
    [
        "attempt_started.json",
        "results.jsonl",
        "candidate.patch",
        "kokkos__kokkos-6375.json",
        "requests",
    ],
)
def test_any_original_partial_artifact_blocks_resampling(tmp_path, filename):
    task, _, kwargs = fixture(tmp_path)
    write(kwargs["original_root"] / MODEL.split("/")[-1] / task.task_name / filename, {})
    with pytest.raises(ValueError, match="never resample"):
        ledger.eligible_evidence(**kwargs)


@pytest.mark.parametrize("kind", ["task", "source", "coverage_blocker", "environment", "resource"])
def test_stale_or_incompatible_qualification_blocks_claim(tmp_path, kind):
    task, source, kwargs = fixture(tmp_path)
    if kind == "task":
        (task.task_dir / "instruction.md").write_text("changed")
    elif kind == "coverage_blocker":
        c = json.loads(kwargs["coverage_path"].read_text())
        c["blockers"] = [{"task": task.task_name}]
        write(kwargs["coverage_path"], c)
    else:
        record = json.loads(source.read_text())
        record[
            "oracle"
            if kind == "source"
            else "runtime_environment_policy_version"
            if kind == "environment"
            else "backend"
        ] = 0 if kind == "source" else "wrong"
        write(source, record)
        if kind != "source":
            q = json.loads(kwargs["qualification_path"].read_text())
            q["tasks"][task.task_name]["successful_evidence"] = [ledger.pinned_file(source)]
            write(kwargs["qualification_path"], q)
    with pytest.raises(ValueError):
        ledger.eligible_evidence(**kwargs)


@pytest.mark.parametrize("change", ["source", "instruction", "artifact", "empty_sources"])
def test_changes_between_eligibility_and_claim_do_not_create_marker(tmp_path, change):
    task, source, kwargs = fixture(tmp_path)
    evidence = ledger.eligible_evidence(**kwargs)
    if change == "source":
        write(source, {})
    elif change == "instruction":
        (task.task_dir / "instruction.md").write_text("changed")
    elif change == "artifact":
        write(
            kwargs["original_root"]
            / MODEL.split("/")[-1]
            / task.task_name
            / "attempt_started.json",
            {},
        )
    else:
        evidence["successful_verifier_evidence"] = []
    with pytest.raises(ValueError):
        ledger.claim_pair(
            task=task,
            ledger_root=tmp_path / "claims",
            phase_root=tmp_path / "new",
            original_root=kwargs["original_root"],
            reserved_original_tasks=kwargs["reserved_original_tasks"],
            evidence=evidence,
            phase_identity_sha256="identity",
        )
    assert not (tmp_path / "claims").exists()


def test_modal_original_gate_requires_matching_resource_and_task_hash(tmp_path):
    task, _, kwargs = fixture(tmp_path)
    old = kwargs["original_root"]
    config = json.loads((old / "launch.json").read_text())
    config["config"].update(
        modal_task_names=[task.task_name], modal_memory_mb=16384, modal_cpus=4.0
    )
    write(old / "launch.json", config)
    path = old / "validation_overrides" / f"{task.task_name}.json"
    value = {
        "passed": True,
        "backend": "modal",
        "memory_mb": 16384,
        "cpu": 4.0,
        "task_hash": "wrong",
    }
    write(path, value)
    assert task.task_name not in ledger.original_gate(old)[0]
    write(path, {**value, "task_hash": ledger._task_digest(task)})
    assert task.task_name in ledger.original_gate(old)[0]


@pytest.mark.parametrize("change", ["missing", "hash", "pending", "empty_sources", "source"])
def test_positive_scope_review_is_required(tmp_path, change):
    task, _, kwargs = fixture(tmp_path)
    path = kwargs["scope_approval_path"]
    data = json.loads(path.read_text())
    row = data["approvals"][task.task_name]
    if change == "missing":
        data["approvals"] = {}
    elif change == "hash":
        row["task_hash"] = "wrong"
    elif change == "pending":
        row["status"] = "pending"
    elif change == "empty_sources":
        row["sources"] = []
    else:
        write(tmp_path / "manual_review.json", {"decision": "changed"})
    write(path, data)
    with pytest.raises((ValueError, TypeError)):
        ledger.eligible_evidence(**kwargs)


@pytest.mark.parametrize("change", ["allowlist", "source"])
def test_scope_proof_mutation_before_claim_blocks_dispatch(tmp_path, change):
    task, _, kwargs = fixture(tmp_path)
    evidence = ledger.eligible_evidence(**kwargs)
    write(
        kwargs["scope_approval_path"] if change == "allowlist" else tmp_path / "manual_review.json",
        {},
    )
    with pytest.raises(ValueError):
        ledger.claim_pair(
            task=task,
            ledger_root=tmp_path / "claims",
            phase_root=tmp_path / "new",
            original_root=kwargs["original_root"],
            reserved_original_tasks=kwargs["reserved_original_tasks"],
            evidence=evidence,
            phase_identity_sha256="identity",
        )
    assert not (tmp_path / "claims").exists()
