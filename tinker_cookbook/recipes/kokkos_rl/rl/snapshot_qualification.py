"""Merge verifier evidence against an immutable snapshot without running sandboxes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import (
    Policy,
    matching_evidence,
    passed_evidence,
    policy_for_task,
)


@chz.chz
class Config:
    snapshot_dir: str
    evidence_dirs: tuple[str, ...]
    output_path: str
    coverage_review_path: str | None = None


def write_report(path: Path, report: dict[str, object]) -> None:
    """Allow independent read-only monitors to refresh the same report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
        json.dump(report, file, indent=2)
        file.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def qualify(config: Config) -> dict[str, object]:
    snapshot = Path(config.snapshot_dir)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = load_harbor_tasks_from_dir(snapshot / "tasks")
    hashes = {task.task_name: _task_digest(task) for task in tasks}
    if not tasks or hashes != manifest["task_hashes"]:
        raise ValueError("Snapshot contents must exactly match a nonempty manifest")
    policies = {task.task_name: policy_for_task(task) for task in tasks}
    evidence: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for directory in config.evidence_dirs:
        folder = Path(directory)
        if not folder.is_dir():
            raise ValueError(f"Evidence directory does not exist: {folder}")
        for path in sorted(folder.glob("*.json")):
            data = path.read_bytes()
            record = json.loads(data)
            if not isinstance(record, dict) or not isinstance(record.get("task"), str):
                raise ValueError(f"Expected a per-task validation record: {path}")
            provenance: dict[str, object] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
                "task_hash": record.get("task_hash"),
                "stage": record.get("stage"),
                "passed": record.get("passed"),
                "nop": record.get("nop"),
                "oracle": record.get("oracle"),
                "policy": {key: record.get(key) for key in asdict(Policy())},
            }
            evidence.setdefault(record["task"], []).append((record, provenance))
    results: dict[str, dict[str, object]] = {}
    for task in tasks:
        name = task.task_name
        matching, rejected = [], []
        for record, provenance in evidence.get(name, []):
            if matching_evidence(record, name, hashes[name], policies[name]):
                matching.append((record, provenance))
            else:
                rejected.append({**provenance, "reason": "task hash or resource policy differs"})
        successful = [source for record, source in matching if passed_evidence(record)]
        results[name] = {
            "task_hash": hashes[name],
            "policy": asdict(policies[name]),
            "qualified": bool(successful),
            "stage": "qualified" if successful else "failed" if matching else "missing",
            "successful_evidence": successful,
            "failed_evidence": [source for record, source in matching if not passed_evidence(record)],
            "rejected_evidence": rejected,
        }
    qualified = sum(result["qualified"] is True for result in results.values())
    coverage_review: dict[str, object] = {"status": "missing", "complete": False}
    if config.coverage_review_path is not None:
        review_path = Path(config.coverage_review_path)
        data = review_path.read_bytes()
        review = json.loads(data)
        same_snapshot = review.get("task_hashes") == hashes
        coverage_review = {
            "path": str(review_path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "status": review.get("status") if same_snapshot else "stale_snapshot",
            "blockers": review.get("blockers"),
            "complete": same_snapshot
            and review.get("status") == "complete"
            and review.get("blockers") == [],
        }
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "snapshot_dir": str(snapshot.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "task_hashes": hashes,
        "total": len(tasks),
        "qualified": qualified,
        "verifier_pairs_passed": qualified,
        "regression_coverage_review": coverage_review,
        "ready_for_sampling": qualified == len(tasks) and coverage_review["complete"] is True,
        "model_requests": 0,
        "tasks": results,
        "unrelated_evidence": {
            name: [provenance for _, provenance in records]
            for name, records in evidence.items()
            if name not in hashes
        },
    }
    write_report(Path(config.output_path), report)
    return report


def main(config: Config) -> None:
    report = qualify(config)
    print(json.dumps({key: report[key] for key in ("total", "qualified", "ready_for_sampling")}))


if __name__ == "__main__":
    chz.entrypoint(main)
