"""Explicit, stage-specific recovery of proven verifier infrastructure failures."""

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
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import grade_patch
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import (
    Policy,
    matching_evidence,
    pid_alive,
)
from tinker_cookbook.sandbox.contree_polling import OperationPollPolicy, create_polling_client
from tinker_cookbook.utils.ml_log import dump_config


@chz.chz
class Config:
    snapshot_dir: str
    proposal_path: str
    output_path: str
    wait_for_pids: tuple[int, ...] = ()
    cache_path: str = "notes/experiments/SWE-kokkos-bench/v2-pass-at-k/contree_images.json"
    env_file: str = ".env"


@dataclass(frozen=True)
class RecoveryPlan:
    task: str
    task_hash: str
    stages: tuple[str, ...]
    reused_nop: float | None
    provenance: dict[str, object]


def plan_record(plan: RecoveryPlan) -> dict[str, object]:
    return {**asdict(plan), "stages": list(plan.stages)}


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected a JSON object")
    return value


def read_json(path: Path) -> dict[str, object]:
    return mapping(json.loads(path.read_text()))


def evidence_file(path: Path, expected_hash: object) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Recovery evidence changed: {path}")
    return {"path": str(path.resolve()), "sha256": digest}


def verify_plan(task: HarborTask, item: dict[str, object]) -> RecoveryPlan:
    digest = _task_digest(task)
    if item.get("task") != task.task_name or item.get("task_hash") != digest:
        raise ValueError("Recovery task identity changed")
    if item.get("resource_policy") != asdict(Policy()):
        raise ValueError("Recovery only supports the reviewed unchanged ConTree resources")
    if item.get("proposed_model_requests") != 0 or item.get("maximum_new_stage_attempts") != 1:
        raise ValueError("Recovery must permit zero inference and one attempt per stage")
    evidence = item.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Recovery requires pinned source evidence")
    files = []
    for value in evidence:
        entry = mapping(value)
        path = entry.get("path")
        if not isinstance(path, str):
            raise ValueError("Evidence path must be a string")
        files.append(evidence_file(Path(path), entry.get("sha256")))
    source_path = Path(str(files[0]["path"]))
    original = read_json(source_path)
    if (
        not matching_evidence(original, task.task_name, digest, Policy())
        or original.get("stage") != "error"
    ):
        raise ValueError("Only original infrastructure errors can enter recovery")
    if original.get("failed_stage") != item.get("original_failed_stage"):
        raise ValueError("Original failed stage differs from recovery proposal")
    source_folder = source_path.parent.parent / task.task_name
    reason = item.get("reason")
    reused_nop = None
    stages = ("nop", "oracle")
    if reason in {
        "original_oracle_cancelled_by_sdk_after_status_transport_error",
        "original_nop_completed_recovered_readonly_oracle_never_submitted",
    }:
        if len(files) < 3:
            raise ValueError("Operation readback and NOP evidence are required")
        operation = read_json(Path(str(files[1]["path"])))
        operation_id = item.get("original_operation_id")
        if not isinstance(operation_id, str) or operation.get("operation_id") != operation_id:
            raise ValueError("Operation identity changed")
        failed_stage = str(original["failed_stage"])
        failed_path = source_folder / f"{failed_stage}.json"
        files.append(evidence_file(failed_path, operation.get("source_sha256")))
        failed = read_json(failed_path)
        error = str(failed.get("stderr", ""))
        if operation_id not in error or not any(
            s in error for s in ("ApiTimeoutError", "ContreeTransportError")
        ):
            raise ValueError("Original error is not a status transport failure for this operation")
        if reason == "original_oracle_cancelled_by_sdk_after_status_transport_error":
            nop_path = source_folder / "nop.json"
            if not any(f["path"] == str(nop_path.resolve()) for f in files):
                raise ValueError("Original completed NOP log is not pinned")
            if (
                failed_stage != "oracle"
                or original.get("nop") != 0
                or read_json(nop_path).get("exit_code") != 0
            ):
                raise ValueError("Original NOP did not complete successfully with reward zero")
            if operation.get("status") != "CANCELLED" or operation.get("result_image"):
                raise ValueError("Only a confirmed cancelled Oracle can be executed again")
        else:
            state = mapping(operation.get("state"))
            if (
                failed_stage != "nop"
                or operation.get("status") != "SUCCESS"
                or not operation.get("result_image")
                or operation.get("reward_text") != "0"
                or state.get("timed_out") is not False
                or state.get("exit_code") != 0
                or (source_folder / "oracle.json").exists()
            ):
                raise ValueError(
                    "Readback does not prove completed NOP zero and an unsubmitted Oracle"
                )
            rewards = [
                Path(str(f["path"])) for f in files if Path(str(f["path"])).name == "reward.txt"
            ]
            if len(rewards) != 1 or rewards[0].read_text().strip() != "0":
                raise ValueError("Immutable NOP reward evidence is missing")
        reused_nop, stages = 0.0, ("oracle",)
    elif reason == "verifier_never_executed_due_to_failed_staging":
        nop_path = source_folder / "nop.json"
        if not any(f["path"] == str(nop_path.resolve()) for f in files):
            raise ValueError("Original staging failure log is not pinned")
        failed = read_json(nop_path)
        if (
            original.get("failed_stage") != "nop"
            or failed.get("exit_code") != 127
            or "/tests/test.sh: No such file or directory" not in str(failed.get("stderr", ""))
            or (source_folder / "oracle.json").exists()
        ):
            raise ValueError("Evidence does not prove the verifier was never executed")
    else:
        raise ValueError("Unreviewed recovery reason")
    if item.get("stages_to_execute") != list(stages) or item.get("reuse_nop") != reused_nop:
        raise ValueError("Proposed stages differ from proven missing work")
    return RecoveryPlan(
        task.task_name,
        digest,
        stages,
        reused_nop,
        {"reason": reason, "files": files, "source_record": original},
    )


async def recover_one(
    task: HarborTask, factory: SandboxFactory, folder: Path, plan: RecoveryPlan, commit: str
) -> dict[str, object]:
    base = {
        "task": task.task_name,
        "task_hash": plan.task_hash,
        **asdict(Policy()),
        "model_requests": 0,
        "recovery_plan": plan_record(plan),
    }
    record: dict[str, object] = {
        **base,
        "code_commit": commit,
        "passed": False,
        "stage": "recovering",
        "nop": plan.reused_nop,
    }
    stage_evidence: dict[str, object] = {}
    record["stage_evidence"] = stage_evidence
    for stage in plan.stages:
        if _task_digest(task) != plan.task_hash:
            raise ValueError("Task changed before recovery stage")
        files = plan.provenance["files"]
        if not isinstance(files, list):
            raise ValueError("Recovery provenance files must be a list")
        for value in files:
            source = mapping(value)
            evidence_file(Path(str(source["path"])), source["sha256"])
        stage_folder = folder / "stages" / stage
        marker, result_path = stage_folder / "attempt_started.json", stage_folder / "result.json"
        identity = {**base, "verifier_stage": stage}
        if result_path.exists():
            result = read_json(result_path)
            if result.get("identity") != identity:
                raise ValueError("Saved recovery stage has another identity")
        elif marker.exists():
            record.update(stage="interrupted", failed_stage=stage)
            write_json(folder / "status.json", record)
            return record
        else:
            write_json(
                marker,
                {"identity": identity, "code_commit": commit, "at": datetime.now(UTC).isoformat()},
            )
            write_json(folder / "status.json", {**record, "stage": stage})
            try:
                patch = (
                    None if stage == "nop" else (task.task_dir / "solution/gold.patch").read_text()
                )
                score = await grade_patch(task, factory, patch, folder / f"{stage}.json")
                result = {"identity": identity, "score": score, "error_type": None}
            except Exception as error:
                result = {"identity": identity, "error_type": type(error).__name__}
            write_json(result_path, result)
        stage_evidence[stage] = [
            {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (marker, result_path, folder / f"{stage}.json")
            if p.exists()
        ]
        if result.get("error_type") is not None:
            record.update(stage="error", failed_stage=stage, error_type=result["error_type"])
            break
        score = result.get("score")
        if score not in (0, 1):
            raise ValueError("Recovery produced an invalid reward")
        record[stage] = score
        if stage == "nop" and score != 0:
            record.update(stage="failed", failed_stage="nop")
            break
    else:
        record["passed"] = record.get("nop") == 0 and record.get("oracle") == 1
        record["stage"] = "complete" if record["passed"] else "failed"
    write_json(folder / "status.json", record)
    return record


async def main(config: Config) -> None:
    root, snapshot = Path(config.output_path), Path(config.snapshot_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = {t.task_name: t for t in load_harbor_tasks_from_dir(snapshot / "tasks")}
        hashes = {n: _task_digest(t) for n, t in tasks.items()}
        if hashes != read_json(snapshot / "manifest.json")["task_hashes"]:
            raise ValueError("Frozen snapshot changed")
        proposal_path = Path(config.proposal_path)
        proposal = read_json(proposal_path)
        if proposal.get("new_transport_policy") != "readonly_status_retry_v1":
            raise ValueError("Recovery requires the reviewed status polling policy")
        entries = proposal.get("tasks")
        if not isinstance(entries, list):
            raise ValueError("Missing recovery task list")
        plans = [verify_plan(tasks[str(mapping(e)["task"])], mapping(e)) for e in entries]
        if len({p.task for p in plans}) != len(plans):
            raise ValueError("Duplicate recovery tasks")
        poll_policy = OperationPollPolicy()
        identity = {
            "config": dump_config(config),
            "proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
            "task_hashes": hashes,
            "plans": [plan_record(p) for p in plans],
            "poll_policy": asdict(poll_policy),
            "model_requests": 0,
            "max_concurrency": 1,
        }
        if (root / "identity.json").exists() and read_json(root / "identity.json") != identity:
            raise ValueError("Recovery identity changed")
        write_json(root / "identity.json", identity)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        process = {
            "pid": os.getpid(),
            "commit": commit,
            "stage": "waiting",
            "started_at": datetime.now(UTC).isoformat(),
        }
        with (root / "launch_history.jsonl").open("a") as stream:
            stream.write(json.dumps(process) + "\n")
        write_json(root / "process.json", process)
        while any(pid_alive(p) for p in config.wait_for_pids):
            await asyncio.sleep(15)
        process["stage"] = "recovering"
        write_json(root / "process.json", process)
        load_env_file(Path(config.env_file))
        factory = ImportedCacheFactory(
            root / "contree_images.json",
            timeout=3600,
            runtime_build_parallelism=1,
            allow_network=False,
        )
        factory._client = create_polling_client(factory._client.config, poll_policy)
        results = {}
        for plan in plans:
            if any(pid_alive(p) for p in config.wait_for_pids):
                raise RuntimeError("Reserved process reappeared; refuse dispatch")
            factory.import_cache(Path(config.cache_path))
            result = await recover_one(tasks[plan.task], factory, root / plan.task, plan, commit)
            write_json(root / "validated" / f"{plan.task}.json", result)
            results[plan.task] = result
            write_json(
                root / "status.json",
                {
                    "stage": "recovering",
                    "finished": len(results),
                    "total": len(plans),
                    "passed": sum(r["passed"] is True for r in results.values()),
                },
            )
        write_json(root / "result.json", results)
        process["stage"] = "complete"
        write_json(root / "process.json", process)
        write_json(
            root / "status.json",
            {
                "stage": "complete",
                "finished": len(results),
                "total": len(plans),
                "passed": sum(r["passed"] is True for r in results.values()),
            },
        )


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(main))
