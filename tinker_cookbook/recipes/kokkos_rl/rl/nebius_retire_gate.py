"""Permanently retire a stopped legacy queue without touching any model attempt."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import original_gate, pinned_file
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_request_audit import write_exclusive
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json


def plan_retirement(original_root: Path, holding: Path) -> Path:
    """Write the complete recovery manifest before moving any permission file."""
    path = holding / "manifest.json"
    if path.exists():
        raise ValueError("A retirement manifest exists; review and resume that exact plan")
    launch = read_json(original_root / "launch.json")
    config = mapping(launch["config"])
    manifest = read_json(Path(str(config["source_manifest"])))
    names = mapping(manifest["task_hashes"])
    entries = []
    for label, folder in (
        ("normal", Path(str(config["validation_dir"]))),
        ("modal", original_root / "validation_overrides"),
    ):
        for name in sorted(names):
            source = folder / f"{name}.json"
            if not source.exists() or read_json(source).get("passed") is not True:
                continue
            destination = holding / "records" / label / source.name
            if destination.exists() or source.is_symlink():
                raise ValueError("Retirement destination exists or source is not a regular record")
            entries.append(
                {
                    "source": pinned_file(source),
                    "bytes": source.stat().st_size,
                    "destination": str(destination.resolve()),
                }
            )
    write_exclusive(
        path,
        {
            "original_root": str(original_root.resolve()),
            "holding": str(holding.resolve()),
            "original_launch": pinned_file(original_root / "launch.json"),
            "source_manifest": pinned_file(Path(str(config["source_manifest"]))),
            "planned_at": datetime.now(UTC).isoformat(),
            "restore_permitted": False,
            "model_requests": 0,
            "entries": entries,
        },
    )
    return path


def _append(path: Path, entry: object) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def retire_planned(path: Path) -> None:
    """Resume interrupted moves, validating original bytes at every boundary."""
    manifest = read_json(path)
    holding = Path(str(manifest["holding"]))
    if path.resolve() != holding / "manifest.json":
        raise ValueError("Retirement manifest is outside its declared holding directory")
    with (holding / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in ("original_launch", "source_manifest"):
            proof = mapping(manifest[key])
            if pinned_file(Path(str(proof["path"]))) != proof:
                raise ValueError("Original controller identity changed before retirement")
        entries = manifest["entries"]
        if not isinstance(entries, list):
            raise ValueError("Retirement entries must be explicit")
        for raw in entries:
            entry = mapping(raw)
            proof = mapping(entry["source"])
            source, destination = Path(str(proof["path"])), Path(str(entry["destination"]))
            for existing in (source, destination):
                if existing.exists() and (
                    existing.is_symlink()
                    or existing.stat().st_size != entry["bytes"]
                    or hashlib.sha256(existing.read_bytes()).hexdigest() != proof["sha256"]
                ):
                    raise ValueError(
                        "A retirement record changed; preserve both paths for diagnosis"
                    )
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if not os.path.samefile(source, destination):
                        raise ValueError("Both paths exist but are not the same interrupted move")
                else:
                    # A hard link creates the destination exclusively and keeps
                    # bytes intact if the process exits before unlinking source.
                    os.link(source, destination, follow_symlinks=False)
                source.unlink()
                state = "moved"
            elif destination.exists():
                state = "verified_already_moved"
            else:
                raise ValueError("Both source and retirement copy are missing")
            _append(
                holding / "move_log.jsonl",
                {
                    "at": datetime.now(UTC).isoformat(),
                    "state": state,
                    "source": str(source),
                    "destination": str(destination),
                    "sha256": proof["sha256"],
                    "bytes": entry["bytes"],
                },
            )
        remaining, _ = original_gate(Path(str(manifest["original_root"])))
        if remaining:
            raise ValueError("New legacy permissions appeared; stop before borrowing capacity")
        _append(
            holding / "verification_log.jsonl",
            {
                "at": datetime.now(UTC).isoformat(),
                "gate_passed": 0,
                "manifest": pinned_file(path),
                "model_requests": 0,
            },
        )
