"""Validate a frozen benchmark snapshot without generating model trajectories."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    SandboxFactory,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import ImportedCacheFactory, write_json
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.resource_sandbox import create_resource_sandbox_factory
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import grade_patch
from tinker_cookbook.utils.ml_log import dump_config


@dataclass(frozen=True)
class Policy:
    backend: str = "contree"
    build_parallelism: int = 1
    grader_timeout: int = 900
    memory_mb: int | None = None
    cpu: float | None = None
    gpu: str | None = None


@chz.chz
class Config:
    snapshot_dir: str
    output_path: str
    reuse_dirs: tuple[str, ...] = ()
    task_names: tuple[str, ...] = ()
    max_concurrency: int = 3
    wait_for_pids: tuple[int, ...] = ()
    cache_path: str = "notes/experiments/SWE-kokkos-bench/v2-pass-at-k/contree_images.json"
    env_file: str = ".env"


def policy_for_task(task: HarborTask) -> Policy:
    name = task.task_name
    if name in {"kokkos__kokkos-8989", "kokkos__kokkos-9147"}:
        policy = Policy("modal", 4, 900, 16384, 4.0, "L4")
    elif name in {"kokkos__kokkos-8164", "kokkos__kokkos-8399", "kokkos__kokkos-8827"}:
        policy = Policy("modal", 4, 900, 16384, 4.0)
    elif name == "kokkos__kokkos-7244":
        policy = Policy("modal", 1, 900, 16384, 4.0)
    else:
        policy = Policy()
    metadata = json.loads((task.task_dir / "metadata.json").read_text()).get("metadata", {})
    if metadata.get("requires_gpu") and policy.gpu is None:
        raise ValueError(f"GPU task {name} has no reviewed GPU resource policy")
    return policy


def matching_evidence(record: dict[str, object], name: str, digest: str, policy: Policy) -> bool:
    return (
        record.get("task") == name
        and record.get("task_hash") == digest
        and all(record.get(key) == value for key, value in asdict(policy).items())
    )


def passed_evidence(record: dict[str, object]) -> bool:
    return record.get("passed") is True and record.get("nop") == 0 and record.get("oracle") == 1


def pid_alive(pid: int) -> bool:
    if pid < 1:
        raise ValueError("Wait PIDs must be positive")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def validate_one(
    task: HarborTask,
    factory: SandboxFactory,
    folder: Path,
    policy: Policy,
    digest: str,
    commit: str,
) -> dict[str, object]:
    if (folder / "attempt_started.json").exists():
        raise RuntimeError("Existing validation attempt requires diagnosis, never silently retry")
    if _task_digest(task) != digest:
        raise ValueError("Task changed before validation")
    record: dict[str, object] = {
        "task": task.task_name,
        "task_hash": digest,
        **asdict(policy),
        "code_commit": commit,
        "model_requests": 0,
        "passed": False,
        "stage": "nop",
    }
    write_json(folder / "attempt_started.json", record)
    write_json(folder / "status.json", record)
    try:
        record["nop"] = await grade_patch(task, factory, None, folder / "nop.json")
        record["stage"] = "oracle"
        write_json(folder / "status.json", record)
        record["oracle"] = await grade_patch(
            task,
            factory,
            (task.task_dir / "solution/gold.patch").read_text(),
            folder / "oracle.json",
        )
        record["passed"] = record["nop"] == 0 and record["oracle"] == 1
        record["stage"] = "complete" if record["passed"] else "failed"
    except Exception as error:
        request = getattr(error, "request", None)
        record.update(
            failed_stage=record["stage"],
            stage="error",
            error_type=type(error).__name__,
            timeout_type=getattr(error, "timeout_type", None),
            endpoint=str(getattr(request, "url", "")).split("?")[0],
        )
    write_json(folder / "status.json", record)
    return record


async def run(config: Config) -> None:
    if not 1 <= config.max_concurrency <= 3:
        raise ValueError("Snapshot validation permits 1–3 shared concurrent sandboxes")
    root, snapshot = Path(config.output_path), Path(config.snapshot_dir)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    all_tasks = load_harbor_tasks_from_dir(snapshot / "tasks")
    hashes = {task.task_name: _task_digest(task) for task in all_tasks}
    if hashes != manifest["task_hashes"]:
        raise ValueError("Frozen snapshot differs from its manifest")
    if set(config.task_names) - hashes.keys():
        raise ValueError("Selected tasks are absent from the snapshot")
    tasks = [
        task for task in all_tasks if not config.task_names or task.task_name in config.task_names
    ]
    policies = {task.task_name: policy_for_task(task) for task in tasks}
    identity = {
        "task_hashes": hashes,
        "resource_policy": {n: asdict(p) for n, p in policies.items()},
        "config": dump_config(config),
        "model_requests": 0,
    }
    identity_path = root / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Validation identity changed; use another output directory")
    write_json(identity_path, identity)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    process = {
        "pid": os.getpid(),
        "commit": commit,
        "stage": "waiting",
        "started_at": datetime.now(UTC).isoformat(),
    }
    with (root / "launch_history.jsonl").open("a") as history:
        history.write(json.dumps(process) + "\n")
    write_json(root / "process.json", process)
    while any(pid_alive(pid) for pid in config.wait_for_pids):
        await asyncio.sleep(15)
    process["stage"] = "validating"
    write_json(root / "process.json", process)
    # Load prior evidence only after its producing processes have exited.
    prior: dict[str, list[tuple[Path, dict[str, object]]]] = {}
    for directory in config.reuse_dirs:
        for path in Path(directory).glob("*.json"):
            record = json.loads(path.read_text())
            if isinstance(record, dict) and isinstance(record.get("task"), str):
                prior.setdefault(record["task"], []).append((path, record))
    results: dict[str, dict[str, object]] = {}
    pending = []
    for task in tasks:
        name, digest, policy = task.task_name, hashes[task.task_name], policies[task.task_name]
        output = root / "validated" / f"{name}.json"
        if output.exists():
            result = json.loads(output.read_text())
            if not matching_evidence(result, name, digest, policy):
                raise ValueError("Saved validation result has another identity")
            results[name] = result
            continue
        if (root / name / "attempt_started.json").exists():
            results[name] = {
                "task": name,
                "task_hash": digest,
                **asdict(policy),
                "passed": False,
                "stage": "interrupted",
            }
            continue
        matches = [
            (p, r) for p, r in prior.get(name, []) if matching_evidence(r, name, digest, policy)
        ]
        if matches:
            path, record = next(((p, r) for p, r in matches if passed_evidence(r)), matches[-1])
            result = {
                **record,
                **asdict(policy),
                "stage": "reused" if passed_evidence(record) else "blocked_prior_validation",
                "evidence_path": str(path.resolve()),
                "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            write_json(output, result)
            results[name] = result
        else:
            pending.append(task)
    write_json(
        root / "dispatch_plan.json",
        {"pending": [t.task_name for t in pending], "existing": results},
    )
    load_env_file(Path(config.env_file))
    primary = ImportedCacheFactory(
        root / "contree_images.json", timeout=3600, runtime_build_parallelism=1, allow_network=False
    )
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def one(task: HarborTask) -> None:
        async with semaphore:
            if any(pid_alive(pid) for pid in config.wait_for_pids):
                raise RuntimeError("A reserved resource process reappeared; refuse more work")
            primary.import_cache(Path(config.cache_path))
            policy = policies[task.task_name]
            factory = (
                primary
                if policy.backend == "contree"
                else create_resource_sandbox_factory(
                    primary,
                    (task.task_name,),
                    memory_mb=policy.memory_mb or 16384,
                    cpu=policy.cpu or 4.0,
                    build_parallelism=policy.build_parallelism,
                    gpu=policy.gpu,
                )
            )
            record = await validate_one(
                task, factory, root / task.task_name, policy, hashes[task.task_name], commit
            )
            results[task.task_name] = record
            write_json(root / "validated" / f"{task.task_name}.json", record)
            write_json(
                root / "status.json",
                {
                    "stage": "validating",
                    "finished": len(results),
                    "total": len(tasks),
                    "passed": sum(passed_evidence(r) for r in results.values()),
                },
            )

    await asyncio.gather(*(one(task) for task in pending))
    write_json(root / "result.json", results)
    process["stage"] = "complete"
    write_json(root / "process.json", process)
    write_json(
        root / "status.json",
        {
            "stage": "complete",
            "finished": len(results),
            "total": len(tasks),
            "passed": sum(passed_evidence(r) for r in results.values()),
        },
    )


async def main(config: Config) -> None:
    root = Path(config.output_path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await run(config)


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(main))
