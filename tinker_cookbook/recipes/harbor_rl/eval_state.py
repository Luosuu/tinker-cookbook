"""Bind resumable evaluation results to their configuration and task contents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask

# These control scheduling, selection, or storage, rather than an individual trial.
_INVOCATION_FIELDS = {
    "output_path",
    "resume_dir",
    "max_concurrency",
    "max_infra_retries",
    "max_tasks",
    "num_samples",
    "pass_at_k",
    "task_names",
    "tasks_dir",
    "env_file",
    "contree_cache_path",
}


def _task_digest(task: HarborTask) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"instruction": task.instruction, "config": task.config}, sort_keys=True
        ).encode()
    )
    for directory in ("environment", "tests"):
        for path in sorted((task.task_dir / directory).rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(task.task_dir)).encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare_eval_state(
    results_dir: Path,
    config: dict[str, object],
    tasks: list[HarborTask],
    *,
    evaluator: str,
) -> None:
    """Validate before creating clients or spending compute; allow task subsets.

    Legacy results lack task digests and cannot be certified for reuse. Keep
    them intact and require a new output directory rather than guessing.
    """
    path = results_dir / "eval_identity.json"
    identity = {
        "version": 1,
        "evaluator": evaluator,
        "config": {
            k: v
            for k, v in config.items()
            if k not in _INVOCATION_FIELDS and not (k == "sandbox_resource_policy" and v is None)
        },
    }
    task_digests = {task.task_name: _task_digest(task) for task in tasks}
    if len(task_digests) != len(tasks):
        raise ValueError("Evaluation task names must be unique")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Resume configuration differs from the saved experiment; use a new directory"
            )
        saved_tasks = saved["tasks"]
        for name, digest in task_digests.items():
            if name in saved_tasks and saved_tasks[name] != digest:
                raise ValueError(f"Resume task content changed for {name}; use a new directory")
        task_digests = {**saved_tasks, **task_digests}
    elif (results_dir / "results.jsonl").exists():
        raise ValueError("Legacy results have no experiment identity; use a new output directory")
    results_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "tasks": task_digests}, indent=2))
    temporary.replace(path)
