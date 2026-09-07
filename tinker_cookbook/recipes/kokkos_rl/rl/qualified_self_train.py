"""Immutable qualification and explicit review boundaries for self-training."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask, SandboxFactory
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import (
    current_task_digest,
    pinned_file,
    scope_approval,
)
from tinker_cookbook.recipes.kokkos_rl.rl.resource_sandbox import create_resource_sandbox_factory
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import (
    matching_evidence,
    passed_evidence,
    policy_for_task,
)
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json


def qualified_evidence(tasks: list[HarborTask], bundle: Path) -> dict[str, object]:
    """Check actual task bytes and every positive review before opening clients."""
    manifest = read_json(bundle / "manifest.json")
    qualification = read_json(bundle / "qualification.json")
    review = read_json(bundle / "coverage_review.json")
    hashes = {task.task_name: current_task_digest(task) for task in tasks}
    if (
        len(hashes) != 100
        or manifest.get("task_hashes") != hashes
        or qualification.get("task_hashes") != hashes
        or review.get("task_hashes") != hashes
        or qualification.get("ready_for_sampling") is not True
        or qualification.get("qualified") != 100
        or review.get("status") != "complete"
        or review.get("blockers") != []
    ):
        raise ValueError("A complete exact-hash qualification and scope review is required")
    version = str(qualification["resource_policy_version"])
    environments = {"kokkos__pykokkos-422": "dockerfile_env_v1"}
    if qualification.get("required_environment_policies") != environments:
        raise ValueError("Qualified runtime environment identity differs")
    policies = {}
    proofs = []
    for task in tasks:
        name, digest = task.task_name, hashes[task.task_name]
        policies[name] = asdict(policy_for_task(task, version))
        approved = scope_approval(bundle / "scope_allowlist.json", name, digest)
        row = mapping(mapping(qualification["tasks"])[name])
        sources = row.get("successful_evidence")
        if row.get("qualified") is not True or not isinstance(sources, list) or not sources:
            raise ValueError("Missing exact successful verifier pair")
        for item in sources:
            source = mapping(item)
            path = Path(str(source["path"]))
            if pinned_file(path)["sha256"] != source.get("sha256"):
                raise ValueError("Qualified verifier evidence changed")
            record = read_json(path)
            if not passed_evidence(record) or not matching_evidence(
                record, name, digest, policy_for_task(task, version)
            ):
                raise ValueError("Verifier task, reward or resource policy differs")
            if (
                name in environments
                and record.get("runtime_environment_policy_version") != environments[name]
            ):
                raise ValueError("Verifier environment policy differs")
            proofs.append(pinned_file(path))
        approval_sources = approved["sources"]
        if not isinstance(approval_sources, list):
            raise ValueError("Invalid scope source list")
        proofs.extend(approval_sources)
    return {
        "bundle": [
            pinned_file(bundle / name)
            for name in (
                "manifest.json",
                "qualification.json",
                "coverage_review.json",
                "scope_allowlist.json",
            )
        ],
        "source_proofs": proofs,
        "resources": policies,
        "resource_policy_version": version,
        "required_environment_policies": environments,
        "operation_poll_policy": "readonly_status_retry_v1",
    }


def qualified_factory(
    primary: SandboxFactory, tasks: list[HarborTask], version: str
) -> SandboxFactory:
    factory = primary
    for task in tasks:
        policy = policy_for_task(task, version)
        if policy.backend == "modal":
            factory = create_resource_sandbox_factory(
                factory,
                (task.task_name,),
                memory_mb=policy.memory_mb or 16384,
                cpu=policy.cpu or 4.0,
                build_parallelism=policy.build_parallelism,
                gpu=policy.gpu,
            )
        elif policy.build_parallelism != 1:
            raise ValueError("Qualified ConTree policy requires build parallelism one")
    return factory


def review_ready(root: Path, label: str, files: list[Path]) -> bool:
    """A review receipt accepts specific immutable artifacts, never a stage name alone."""
    if not files or any(not path.is_file() for path in files):
        raise ValueError("Review artifacts are missing")
    evidence = {"stage": label, "files": [pinned_file(path) for path in sorted(files)]}
    required = root / f"{label}_review_required.json"
    if required.exists() and read_json(required) != evidence:
        raise ValueError("Previously reviewed artifacts changed")
    if not required.exists():
        required.write_text(json.dumps(evidence, indent=2) + "\n")
    receipt = root / f"{label}_review.json"
    if not receipt.exists():
        return False
    accepted = read_json(receipt)
    if accepted.get("status") != "accepted" or accepted.get("evidence") != pinned_file(required):
        raise ValueError("Review receipt does not accept this exact evidence")
    return True
