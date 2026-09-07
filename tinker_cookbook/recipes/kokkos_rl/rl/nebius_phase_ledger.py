"""Read-only eligibility checks and exclusive claims across a benchmark's phases.

The original controller does not consult this ledger. Reserve its entire live
queue, including unattempted tasks, and require that queue to only shrink. This
module never starts a model request, resumes a partial attempt, or edits a peer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import MODELS
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import (
    Policy,
    matching_evidence,
    passed_evidence,
)
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json


class OriginalGateExpandedError(ValueError):
    """A peer can now schedule new pairs; stop dispatch until coordination is restored."""


def pinned_file(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def original_gate(phase_root: Path) -> tuple[frozenset[str], list[dict[str, str]]]:
    config = mapping(read_json(phase_root / "launch.json")["config"])
    manifest = read_json(Path(str(config["source_manifest"])))
    names = mapping(manifest["task_hashes"])
    modal_names = config["modal_task_names"]
    if not isinstance(modal_names, list):
        raise ValueError("Original Modal task mapping must be explicit")
    reserved, evidence = set(), []
    for name in names:
        modal = name in modal_names
        folder = (
            phase_root / "validation_overrides" if modal else Path(str(config["validation_dir"]))
        )
        path = folder / f"{name}.json"
        if not path.exists():
            continue
        record = read_json(path)
        passed = record.get("passed") is True
        if modal:
            passed = passed and (
                record.get("backend") == "modal"
                and record.get("memory_mb") == config["modal_memory_mb"]
                and record.get("cpu") == config["modal_cpus"]
                and record.get("task_hash") == names[name]
            )
        if passed:
            reserved.add(name)
            evidence.append(pinned_file(path))
    return frozenset(reserved), evidence


def require_nonexpanding_original_gate(phase_root: Path, reserved: frozenset[str]) -> None:
    current, _ = original_gate(phase_root)
    if current - reserved:
        raise OriginalGateExpandedError("Original controller gate expanded; stop new dispatch")


def attempt_artifacts(phase_root: Path, model: str, name: str) -> list[Path]:
    if model not in MODELS or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Unrecognized benchmark model or task name")
    folder = phase_root / model.split("/")[-1] / name
    paths = [
        folder / "attempt_started.json",
        folder / f"{name}.json",
        folder / "results.jsonl",
        folder / "candidate.json",
        folder / "candidate.patch",
    ]
    if (folder / "requests").exists():
        paths.append(folder / "requests")
    return [p for p in paths if p.exists()]


def current_task_digest(task: HarborTask) -> str:
    if (task.task_dir / "instruction.md").read_text() != task.instruction or tomllib.loads(
        (task.task_dir / "task.toml").read_text()
    ) != task.config:
        raise ValueError("Frozen task instruction or configuration changed on disk")
    return _task_digest(task)


def scope_approval(path: Path, name: str, digest: str) -> dict[str, object]:
    """Require a positive, explicitly accepted review with immutable source proof."""
    approvals = mapping(read_json(path).get("approvals"))
    approval = mapping(approvals.get(name))
    if (
        approval.get("task") != name
        or approval.get("task_hash") != digest
        or approval.get("status") != "accepted"
    ):
        raise ValueError("Missing accepted scope approval for this exact task")
    sources = approval.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Scope approval requires original review evidence")
    verified = []
    for value in sources:
        proof = mapping(value)
        if pinned_file(Path(str(proof["path"]))) != proof:
            raise ValueError("Scope review source changed after approval")
        verified.append(proof)
    return {"allowlist": pinned_file(path), "approval": approval, "sources": verified}


def eligible_evidence(
    *,
    task: HarborTask,
    model: str,
    original_root: Path,
    reserved_original_tasks: frozenset[str],
    qualification_path: Path,
    coverage_path: Path,
    scope_approval_path: Path,
    policy: Policy,
    required_environment_policy: str | None = None,
) -> dict[str, object]:
    require_nonexpanding_original_gate(original_root, reserved_original_tasks)
    name = task.task_name
    if name in reserved_original_tasks:
        raise ValueError("Task is reserved for the original controller, even without an attempt")
    if attempt_artifacts(original_root, model, name):
        raise ValueError("Original attempt or partial artifacts exist; never resample")
    digest = current_task_digest(task)
    qualification, coverage = read_json(qualification_path), read_json(coverage_path)
    hashes = mapping(qualification["task_hashes"])
    if hashes.get(name) != digest or coverage.get("task_hashes") != hashes:
        raise ValueError("Qualification, coverage, and actual task hashes must match")
    if coverage.get("status") not in {"pending", "complete"}:
        raise ValueError("Coverage review state does not permit staged evaluation")
    blockers = coverage.get("blockers")
    if not isinstance(blockers, list) or any(mapping(b).get("task") == name for b in blockers):
        raise ValueError("Task coverage review is not ready")
    approved_scope = scope_approval(scope_approval_path, name, digest)
    row = mapping(mapping(qualification["tasks"])[name])
    sources = row.get("successful_evidence")
    if row.get("qualified") is not True or not isinstance(sources, list) or not sources:
        raise ValueError("Task requires an exact successful verifier pair")
    verified = []
    for value in sources:
        source = mapping(value)
        path = Path(str(source["path"]))
        proof = pinned_file(path)
        if proof["sha256"] != source.get("sha256"):
            raise ValueError("Verifier evidence changed after qualification")
        record = read_json(path)
        if not matching_evidence(record, name, digest, policy) or not passed_evidence(record):
            raise ValueError("Verifier payload or resource policy differs")
        if (
            required_environment_policy is not None
            and record.get("runtime_environment_policy_version") != required_environment_policy
        ):
            raise ValueError("Verifier runtime environment policy differs")
        verified.append(proof)
    return {
        "model": model,
        "task": name,
        "task_hash": digest,
        "qualification": pinned_file(qualification_path),
        "coverage_review": pinned_file(coverage_path),
        "scope_approval": approved_scope,
        "successful_verifier_evidence": verified,
        "required_environment_policy": required_environment_policy,
    }


def claim_pair(
    *,
    task: HarborTask,
    ledger_root: Path,
    phase_root: Path,
    original_root: Path,
    reserved_original_tasks: frozenset[str],
    evidence: dict[str, object],
    phase_identity_sha256: str,
) -> Path:
    """Persist ownership before any request; an existing claim always blocks reuse."""
    model, name = str(evidence["model"]), str(evidence["task"])
    if task.task_name != name or current_task_digest(task) != evidence.get("task_hash"):
        raise ValueError("Actual task changed before claim")
    require_nonexpanding_original_gate(original_root, reserved_original_tasks)
    if name in reserved_original_tasks or attempt_artifacts(original_root, model, name):
        raise ValueError("Original controller owns or has attempted this pair")
    if attempt_artifacts(phase_root, model, name):
        raise ValueError("This phase already has attempt artifacts; do not repeat")
    for key in ("qualification", "coverage_review"):
        proof = mapping(evidence[key])
        if pinned_file(Path(str(proof["path"]))) != proof:
            raise ValueError("Eligibility evidence changed before claim")
    scope = mapping(evidence["scope_approval"])
    allowlist = mapping(scope["allowlist"])
    if (
        pinned_file(Path(str(allowlist["path"]))) != allowlist
        or scope_approval(Path(str(allowlist["path"])), name, str(evidence["task_hash"])) != scope
    ):
        raise ValueError("Scope approval changed before claim")
    sources = evidence["successful_verifier_evidence"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("Missing verifier evidence")
    for source in sources:
        proof = mapping(source)
        if pinned_file(Path(str(proof["path"]))) != proof:
            raise ValueError("Verifier evidence changed before claim")
    path = ledger_root / model.split("/")[-1] / name / "claim.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "phase_root": str(phase_root.resolve()),
        "phase_identity_sha256": phase_identity_sha256,
        "claimed_at": datetime.now(UTC).isoformat(),
        "evidence": evidence,
        "status": "claimed_before_model_request",
    }
    # Never replace or clear this file, including an incomplete write after a crash.
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def original_model_capacity(
    original_root: Path,
    reserved_original_tasks: frozenset[str],
    *,
    launch_proof: dict[str, str],
    manifest_proof: dict[str, str],
    now: datetime,
    max_status_age_seconds: float = 90,
) -> dict[str, object]:
    """Only lend slots once the entire reserved queue is terminal or in flight.

    Markers without terminal results remain in flight even when absent from the
    status list. A finished response still listed active remains reserved until
    the original controller publishes that its task (including cleanup) ended.
    """
    if max_status_age_seconds <= 0 or max_status_age_seconds > 90:
        raise ValueError("Status freshness must be within ninety seconds")
    for proof in (launch_proof, manifest_proof):
        if pinned_file(Path(proof["path"])) != proof:
            raise ValueError("Original controller identity changed")
    if Path(launch_proof["path"]).resolve() != (original_root / "launch.json").resolve():
        raise ValueError("Wrong original launch proof")
    launch = read_json(original_root / "launch.json")
    config = mapping(launch["config"])
    if Path(str(config["source_manifest"])).resolve() != Path(manifest_proof["path"]).resolve():
        raise ValueError("Wrong original manifest proof")
    if config.get("max_concurrency") != 4:
        raise ValueError("Original shared model limit must remain four")
    require_nonexpanding_original_gate(original_root, reserved_original_tasks)
    status = read_json(original_root / "status.json")
    age = (now - datetime.fromisoformat(str(status["updated_at"]))).total_seconds()
    if not 0 <= age <= max_status_age_seconds:
        raise ValueError("Original controller status is stale")
    active = status.get("active")
    if not isinstance(active, list):
        raise ValueError("Missing original active request accounting")
    outstanding: set[tuple[str, str]] = set()
    for pair in active:
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] not in MODELS:
            raise ValueError("Invalid original active pair")
        outstanding.add((str(pair[0]), str(pair[1])))
    started: set[tuple[str, str]] = set()
    for model in MODELS:
        directory = original_root / model.split("/")[-1]
        for marker in directory.glob("*/attempt_started.json"):
            name = marker.parent.name
            pair = (model, name)
            started.add(pair)
            result_path = marker.parent / "results.jsonl"
            if not result_path.exists():
                outstanding.add(pair)
                continue
            rows = [
                json.loads(line) for line in result_path.read_text().splitlines() if line.strip()
            ]
            if len(rows) != 1 or mapping(rows[0]).get("task_name") != name:
                raise ValueError("Ambiguous original terminal result")
    if any(pair not in started for pair in outstanding):
        raise ValueError("Original active request is missing its attempt marker")
    pending = {(model, name) for model in MODELS for name in reserved_original_tasks} - started
    free = max(0, 4 - len(outstanding)) if not pending else 0
    return {
        "available_model_slots": free,
        "original_in_flight": sorted(outstanding),
        "reserved_unstarted_pairs": len(pending),
        "status": pinned_file(original_root / "status.json"),
        "status_age_seconds": age,
        "unknown_responses_remain_in_flight": True,
    }
