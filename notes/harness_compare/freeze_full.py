"""Freeze the validated 100-task Rivanna snapshot for four Claude Code samples."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(
    source: Path, destination: Path, code: Path, runtime: Path,
    continuation_policy: str = "none", max_continuations: int = 0,
) -> dict:
    if continuation_policy not in {"none", "thinking_only_resume"}:
        raise ValueError(f"Unknown continuation policy: {continuation_policy}")
    if max_continuations < 0 or (continuation_policy == "none" and max_continuations != 0):
        raise ValueError("Invalid continuation limit")
    if continuation_policy == "thinking_only_resume" and max_continuations == 0:
        raise ValueError("Continuation profile requires a positive limit")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an experiment: {destination}")
    source = source.resolve(strict=True)
    code = code.resolve(strict=True)
    runtime = runtime.resolve(strict=True)
    manifest_path = source / "inputs/manifest.json"
    runtime_manifest_path = source / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runtime_rows = json.loads(runtime_manifest_path.read_text())
    train, heldout = set(manifest["train"]), set(manifest["heldout"])
    names = sorted(train | heldout)
    if len(train) != 80 or len(heldout) != 20 or train & heldout or len(names) != 100:
        raise ValueError("Expected disjoint 80/20 split covering exactly 100 tasks")
    runtime_by_task = {row["task"]: row for row in runtime_rows}
    if len(runtime_by_task) != 100 or set(runtime_by_task) != set(names):
        raise ValueError("Runtime manifest does not cover exactly the 100 split tasks")
    validation = {}
    for group in ("cpu", "gpu"):
        for row in json.loads((source / f"validation-{group}/summary.json").read_text()):
            if row.get("error") or row.get("cleanup_error"):
                raise ValueError(f"Validation infrastructure error for {row['task']}")
            validation.setdefault(row["task"], []).append(row)
    if set(validation) != set(names):
        raise ValueError(f"Missing validation: {sorted(set(names) - set(validation))}")

    with tempfile.TemporaryDirectory(prefix="claude-pass4-freeze-", dir=destination.parent) as tmp:
        staging = Path(tmp) / destination.name
        tasks_dir = staging / "tasks"
        tasks_dir.mkdir(parents=True)
        for name in names:
            shutil.copytree(source / "runtime-tasks" / name, tasks_dir / name)
        task_objects = {task.task_name: task for task in load_harbor_tasks_from_dir(tasks_dir)}
        if set(task_objects) != set(names):
            raise ValueError("Copied task set differs from manifest")

        code_dir = staging / "code"
        code_dir.mkdir()
        code_files = (
            "eval_claude.py", "completion_quality.py", "continuation.py",
            "mcp_hpc.py", "analyze_full.py",
        )
        for filename in code_files:
            shutil.copy2(code / filename, code_dir / filename)
        (code_dir / "claude-runtime").symlink_to(runtime, target_is_directory=True)

        images = {}
        digests = {}
        for index, name in enumerate(names, 1):
            task = task_objects[name]
            row = runtime_by_task[name]
            image = Path((task.task_dir / "environment/sif.path").read_text().strip())
            if image.resolve(strict=True) != Path(row["sif"]).resolve(strict=True):
                raise ValueError(f"SIF path differs for {name}")
            source_dockerfile = source / "inputs/tasks" / name / "environment/Dockerfile"
            if sha256(source_dockerfile) != row["source_dockerfile_sha256"]:
                raise ValueError(f"Original Dockerfile differs for {name}")
            runtime_dockerfile_hash = sha256(task.task_dir / "environment/Dockerfile")
            if not row["cuda"] and runtime_dockerfile_hash != row["source_dockerfile_sha256"]:
                raise ValueError(f"Unexpected CPU Dockerfile rewrite for {name}")
            pair = validation[name]
            if len(pair) != 2 or {v["kind"]: v["reward"] for v in pair} != {"nop": 0.0, "oracle": 1.0}:
                raise ValueError(f"NOP/oracle pair is invalid for {name}")
            if not all(v.get("baseline_ok") and v["sif_sha256"] == row["sif_sha256"] for v in pair):
                raise ValueError(f"SIF validation failed for {name}")
            stat_before = image.stat()
            actual_hash = sha256(image)
            stat_after = image.stat()
            if actual_hash != row["sif_sha256"] or (
                stat_before.st_size, stat_before.st_mtime_ns
            ) != (stat_after.st_size, stat_after.st_mtime_ns):
                raise ValueError(f"SIF changed or hash differs for {name}")
            images[name] = {
                "path": str(image),
                "sha256": actual_hash,
                "size": stat_after.st_size,
                "mtime_ns": stat_after.st_mtime_ns,
                "gpu": bool(row["cuda"]),
                "source_dockerfile_sha256": row["source_dockerfile_sha256"],
                "runtime_dockerfile_sha256": runtime_dockerfile_hash,
            }
            digests[name] = _task_digest(task)
            if index % 10 == 0:
                print(f"Verified {index}/100 task images", flush=True)

        cpu = [name for name in names if not images[name]["gpu"]]
        gpu = [name for name in names if images[name]["gpu"]]
        if len(cpu) != 96 or len(gpu) != 4:
            raise ValueError("Expected 96 CPU and 4 GPU tasks")
        for group, task_names in (("cpu", cpu), ("gpu", gpu)):
            index = [
                {"task": name, "sample_index": sample, "split": "train" if name in train else "heldout"}
                for name in task_names for sample in range(4)
            ]
            (staging / f"{group}-index.json").write_text(json.dumps(index, indent=2))
        design = {
            "created_at": datetime.now(UTC).isoformat(),
            "source": str(source),
            "source_manifest_sha256": sha256(manifest_path),
            "runtime_manifest_sha256": sha256(runtime_manifest_path),
            "tasks": names,
            "splits": {"train": sorted(train), "heldout": sorted(heldout)},
            "samples": 4,
            "model": "thinkingmachines/Inkling-Small:peft:262144",
            "weights": "hosted_chat_not_rft_checkpoint",
            "harness": "claude-code",
            "harness_version": "2.1.50",
            "endpoint": "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api",
            "max_model_turns": 40,
            "max_tool_calls": 80,
            "max_wall_seconds": 3600,
            "continuation_policy": continuation_policy,
            "max_continuations": max_continuations,
            "image": images,
            "task_digest": digests,
            "validation": validation,
            "code_sha256": {name: sha256(code_dir / name) for name in code_files},
            "claude_package_lock_sha256": sha256(runtime / "package-lock.json"),
        }
        (staging / "design.json").write_text(json.dumps(design, indent=2))
        (staging / "results").mkdir()
        (staging / "slurm").mkdir()
        staging.rename(destination)
    return {"tasks": len(names), "cpu": len(cpu), "gpu": len(gpu), "episodes": 400}


if __name__ == "__main__":
    print(json.dumps(freeze(
        *(Path(value) for value in sys.argv[1:5]),
        continuation_policy=sys.argv[5] if len(sys.argv) > 5 else "none",
        max_continuations=int(sys.argv[6]) if len(sys.argv) > 6 else 0,
    )))
