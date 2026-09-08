"""Once-only fresh-sandbox qualification of audited self-training demonstrations.

This runner never constructs an inference or training client. Existing collection
scores are immutable; qualification is an independent donor eligibility check.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import subprocess
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    SandboxFactory,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import ImportedCacheFactory, write_json
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import (
    current_task_digest,
    pinned_file,
)
from tinker_cookbook.recipes.kokkos_rl.rl.qualified_self_train import (
    qualified_evidence,
    qualified_factory,
)
from tinker_cookbook.recipes.kokkos_rl.rl.saved_candidate_regrade import digest, exclusive_json
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import grade_patch
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import pid_alive
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json
from tinker_cookbook.sandbox.contree_polling import OperationPollPolicy, create_polling_client


def records(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("Expected records")
    return [mapping(row) for row in value]


def names(value: object) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected explicit task names")
    return set(value)


def proof_path(proof: dict[str, object]) -> Path:
    path = Path(str(proof["path"])).resolve()
    if pinned_file(path) != proof:
        raise ValueError(f"Pinned source changed: {path}")
    return path


def code_identity() -> dict[str, str]:
    # Pin complete modules, including scheduling and transitively used guards.
    repo = Path(__file__).resolve().parents[4]
    names = [
        "recipes/kokkos_rl/rl/donor_qualification.py",
        "recipes/kokkos_rl/rl/self_train.py",
        "recipes/kokkos_rl/rl/qualified_self_train.py",
        "recipes/kokkos_rl/rl/saved_candidate_regrade.py",
        "recipes/kokkos_rl/rl/nebius_phase_ledger.py",
        "recipes/kokkos_rl/rl/validation_recovery.py",
        "recipes/kokkos_rl/rl/validate_snapshot.py",
        "recipes/kokkos_rl/rl/resource_sandbox.py",
        "recipes/kokkos_rl/rl/eval_kokkos_nebius.py",
        "recipes/kokkos_rl/rl/eval_kokkos_openai.py",
        "recipes/harbor_rl/harbor_tools.py",
        "recipes/harbor_rl/harbor_env.py",
        "sandbox/contree_polling.py",
        "sandbox/contree_sandbox.py",
        "sandbox/modal_sandbox.py",
        "sandbox/sandbox_interface.py",
        "sandbox/__init__.py",
        "tool_use/tools.py",
        "tool_use/types.py",
    ]
    return {name: pinned_file(repo / "tinker_cookbook" / name)["sha256"] for name in names}


def audited_source(audit: dict[str, object], root: Path, source: Path) -> Path:
    proof = mapping(mapping(audit["source_proofs"]).get(str(source.resolve())))
    original = source.read_bytes()
    obj = (root / str(proof["object"])).resolve()
    if not obj.is_relative_to(root.resolve() / "objects"):
        raise ValueError("Audit object escapes archive")
    raw = obj.read_bytes()
    if (
        raw != original
        or len(raw) != proof["bytes"]
        or hashlib.sha256(raw).hexdigest() != proof["sha256"]
    ):
        raise ValueError("Audited original/frozen bytes changed")
    return obj


def validate_candidate(
    row: dict[str, object], task: HarborTask, train: set[str], heldout: set[str]
) -> None:
    name, sample = str(row["task"]), row["sample"]
    if name not in train or name in heldout or type(sample) is not int or sample not in range(4):
        raise ValueError("Donor split/sample mismatch")
    if row["technical_pass"] is not True or row["errors"] != []:
        raise ValueError("Candidate lacks positive technical audit")
    if current_task_digest(task) != row["task_hash"]:
        raise ValueError("Task changed")
    meta = read_json(proof_path(mapping(row["candidate_proof"])))
    patch = proof_path(mapping(row["patch_proof"])).read_bytes()
    alias = proof_path(mapping(row["alias_proof"])).read_bytes()
    trajectory = read_json(proof_path(mapping(row["trajectory_proof"])))
    base = read_json(task.task_dir / "metadata.json")["base_commit"]
    if (
        meta.get("complete") is not True
        or meta.get("head_commit") != base
        or meta.get("base_commit") != base
    ):
        raise ValueError("Candidate is incomplete or wrong HEAD")
    if (
        not patch
        or patch != alias
        or len(patch) != meta["patch_bytes"]
        or hashlib.sha256(patch).hexdigest() != meta["patch_sha256"]
    ):
        raise ValueError("Candidate patch changed")
    if (
        trajectory.get("task_name") != name
        or trajectory.get("sample_index") != sample
        or trajectory.get("reward") != 1
        or trajectory.get("stop_reason") != "completed"
        or trajectory.get("audit_flags") != []
        or not trajectory.get("datums")
        or not 0 < int(str(trajectory["turns"])) <= 40
    ):
        raise ValueError("Not a naturally completed unflagged positive")
    for item in records(row["original_proofs"]):
        proof_path(item)


def prepare(collection: Path, audit_path: Path) -> tuple[dict[str, object], dict[str, HarborTask]]:
    audit = read_json(audit_path)
    if (
        audit.get("candidate_count") != 91
        or audit.get("technical_pass") != 91
        or audit.get("task_coverage") != 31
    ):
        raise ValueError("Expected exact reviewed 91-candidate audit")
    for source in mapping(audit["source_proofs"]):
        audited_source(audit, audit_path.parent, Path(source))
    manifest = read_json(collection / "manifest.json")
    tasks = load_harbor_tasks_from_dir(collection / "tasks")
    task_map = {task.task_name: task for task in tasks}
    if {n: current_task_digest(t) for n, t in task_map.items()} != manifest["task_hashes"]:
        raise ValueError("Original complete task snapshot changed")
    config = read_json(collection / "config.json")
    qualified = qualified_evidence(tasks, Path(str(config["qualified_bundle_dir"])))
    if qualified != manifest["qualification"]:
        raise ValueError("Qualification differs from original collection")
    train, heldout = names(manifest["train"]), names(manifest["heldout"])
    if len(train) != 80 or len(heldout) != 20 or train & heldout:
        raise ValueError("Invalid original split")
    rows = []
    for candidate in records(audit["candidates"]):
        row = dict(candidate)
        name = str(row["task"])
        folder = collection / "collection/rollouts" / str(row["slot"])
        if folder.name != f"{name}__{int(str(row['sample'])):02d}":
            raise ValueError("Slot name differs")
        row["task_hash"] = mapping(manifest["task_hashes"])[name]
        originals = []
        for key, filename in [
            ("candidate_proof", "candidate.json"),
            ("patch_proof", "candidate.patch"),
            ("alias_proof", "patch.diff"),
            ("trajectory_proof", "trajectory.json"),
        ]:
            original = folder / filename
            row[key] = pinned_file(audited_source(audit, audit_path.parent, original))
            originals.append(pinned_file(original))
        row["original_proofs"] = originals
        validate_candidate(row, task_map[name], train, heldout)
        rows.append(row)
    slots = [str(row["slot"]) for row in rows]
    if len(slots) != 91 or len(set(slots)) != 91:
        raise ValueError("Duplicate or missing donor slots")
    return {
        "audit": pinned_file(audit_path),
        "collection_manifest": pinned_file(collection / "manifest.json"),
        "collection_config": pinned_file(collection / "config.json"),
        "original_results": pinned_file(collection / "collection/results.jsonl"),
        "train": sorted(train),
        "heldout": sorted(heldout),
        "qualification": qualified,
        "records": rows,
    }, task_map


def slot_state(folder: Path, claim: dict[str, object]) -> dict[str, object] | None:
    marker, result = folder / "claim.json", folder / "result.json"
    if marker.exists():
        if read_json(marker) != claim:
            raise ValueError("Existing claim identity differs")
        if not result.exists():
            raise ValueError("Claimed partial must not be regraded")
        value = read_json(result)
        if value.get("claim") != claim or value.get("stage") not in ("complete", "infra_error"):
            raise ValueError("Result differs from claim")
        if value.get("stage") == "infra_error" or value.get("error") is not None:
            raise ValueError("Prior qualification error requires review; no automatic continuation")
        if value.get("reward") not in (0, 1):
            raise ValueError("Invalid qualification score")
        return value
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Unclaimed partial artifacts")
    return None


async def grade_once(
    row: dict[str, object],
    task: HarborTask,
    factory: SandboxFactory,
    output: Path,
    identity_sha: str,
    after_grade: Callable[[], None] | None = None,
) -> dict[str, object]:
    folder = output / str(row["slot"])
    claim = {
        "identity_sha256": identity_sha,
        "slot": row["slot"],
        "task_hash": row["task_hash"],
        "patch_sha256": mapping(row["patch_proof"])["sha256"],
        "model_requests": 0,
        "retry": False,
    }
    prior = slot_state(folder, claim)
    if prior is not None:
        return prior
    exclusive_json(folder / "claim.json", claim)
    result: dict[str, object] = {
        "claim": claim,
        "model_requests": 0,
        "outage_review": row["outage_scope_decision"],
        "original_score_unchanged": True,
    }
    try:
        reward = await grade_patch(
            task,
            factory,
            proof_path(mapping(row["patch_proof"])).read_text(),
            folder / "grading.json",
        )
        if after_grade is not None:
            after_grade()
        if reward not in (0, 1):
            raise ValueError("Invalid verifier reward")
        result.update(stage="complete", reward=reward, error=None)
    except BaseException as error:
        result.update(
            stage="infra_error",
            reward=None,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        exclusive_json(folder / "result.json", result)
        raise
    exclusive_json(folder / "result.json", result)
    return result


async def dispatch(
    rows: list[dict[str, object]],
    worker: Callable[[dict[str, object]], Awaitable[dict[str, object]]],
    concurrency: int,
    progress: Callable[[list[dict[str, object]]], None],
) -> list[dict[str, object]]:
    if not 1 <= concurrency <= 4:
        raise ValueError("At most four sandbox slots")
    semaphore = asyncio.Semaphore(concurrency)
    stopped = asyncio.Event()
    results: list[dict[str, object]] = []

    async def one(row: dict[str, object]) -> None:
        async with semaphore:
            if stopped.is_set():
                return
            try:
                result = await worker(row)
            except BaseException as error:
                stopped.set()
                result = {
                    "slot": row["slot"],
                    "stage": "guard_or_infra_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            try:
                progress(results)
            except BaseException as error:
                stopped.set()
                results.append(
                    {"stage": "progress_error", "error": f"{type(error).__name__}: {error}"}
                )

    await asyncio.gather(*(one(row) for row in rows))
    return results


@chz.chz
class Config:
    collection_dir: str
    audit_path: str
    output_dir: str
    max_concurrency: int = 4
    dispatch: bool = False
    review_receipt: str | None = None
    native_pid: int = 26679
    paused_pids: tuple[int, ...] = (6896, 90888)
    env_file: str = ".env"


def capacity(config: Config) -> None:
    if pid_alive(config.native_pid):
        raise ValueError("Original collection process is still alive")
    for pid in config.paused_pids:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], text=True, capture_output=True, check=False
        )
        if result.returncode != 0 or not result.stdout.strip().startswith("T"):
            raise ValueError("Reserved Nebius process is no longer paused")


async def run(config: Config) -> None:
    if not 1 <= config.max_concurrency <= 4:
        raise ValueError("At most four sandbox slots")
    collection, audit_path, output = (
        Path(config.collection_dir).resolve(),
        Path(config.audit_path).resolve(),
        Path(config.output_dir).resolve(),
    )
    if (
        output == collection
        or output.is_relative_to(collection / "collection")
        or output == audit_path.parent
    ):
        raise ValueError("Output must not overlap original artifacts")
    prepared, tasks = prepare(collection, audit_path)
    reservations = {}
    for pid in config.paused_pids:
        reservations[str(pid)] = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "lstart=,command="], text=True
        ).strip()
    identity = {
        "inputs": prepared,
        "execution_sources": code_identity(),
        "concurrency": config.max_concurrency,
        "policy": asdict(OperationPollPolicy()),
        "sandbox_reservation": {"this_phase": 4, "other_reserved": 4, "total": 8},
        "paused_pids": list(config.paused_pids),
        "paused_process_identities": reservations,
        "native_pid": config.native_pid,
        "model_requests": 0,
        "automatic_training": False,
    }
    identity_sha = digest(identity)
    rows = records(prepared["records"])
    for row in rows:
        slot_state(
            output / str(row["slot"]),
            {
                "identity_sha256": identity_sha,
                "slot": row["slot"],
                "task_hash": row["task_hash"],
                "patch_sha256": mapping(row["patch_proof"])["sha256"],
                "model_requests": 0,
                "retry": False,
            },
        )
    capacity(config)
    if not config.dispatch:
        print(
            json.dumps(
                {
                    "identity_sha256": identity_sha,
                    "candidates": len(rows),
                    "concurrency": config.max_concurrency,
                    "remote_requests": 0,
                    "identity": identity,
                }
            )
        )
        return
    if config.review_receipt is None or read_json(Path(config.review_receipt)) != {
        "status": "accepted",
        "identity_sha256": identity_sha,
    }:
        raise ValueError("Exact parent review receipt required before opening sandboxes")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "identity.json").exists():
            if read_json(output / "identity.json") != identity:
                raise ValueError("Phase identity changed")
        else:
            exclusive_json(output / "identity.json", identity)
        load_env_file(Path(config.env_file))
        primary = ImportedCacheFactory(
            output / "contree_images.json",
            timeout=3600,
            runtime_build_parallelism=1,
            allow_network=False,
        )
        primary._client = create_polling_client(primary._client.config, OperationPollPolicy())
        primary.import_cache(collection / "contree_images.json")
        factory = qualified_factory(
            primary,
            list(tasks.values()),
            str(mapping(prepared["qualification"])["resource_policy_version"]),
        )
        write_json(
            output / "process.json",
            {"pid": os.getpid(), "identity_sha256": identity_sha, "stage": "running"},
        )

        def progress(results: list[dict[str, object]]) -> None:
            write_json(
                output / "status.json",
                {
                    "total": 91,
                    "finished": len(results),
                    "passed": sum(r.get("reward") == 1 for r in results),
                    "errors": sum(r.get("stage") != "complete" for r in results),
                    "stage": "running",
                },
            )

        def guard(row: dict[str, object]) -> None:
            capacity(config)
            for pid in config.paused_pids:
                actual = subprocess.check_output(
                    ["ps", "-p", str(pid), "-o", "lstart=,command="], text=True
                ).strip()
                if actual != reservations[str(pid)]:
                    raise ValueError("Paused process identity changed")
            if code_identity() != identity["execution_sources"]:
                raise ValueError("Execution source changed")
            for key in ("audit", "collection_manifest", "collection_config", "original_results"):
                proof_path(mapping(prepared[key]))
            q = mapping(prepared["qualification"])
            for key in ("bundle", "source_proofs"):
                for proof in records(q[key]):
                    proof_path(proof)
            validate_candidate(
                row, tasks[str(row["task"])], names(prepared["train"]), names(prepared["heldout"])
            )

        async def one(row: dict[str, object]) -> dict[str, object]:
            guard(row)
            return await grade_once(
                row, tasks[str(row["task"])], factory, output, identity_sha, lambda: guard(row)
            )

        results = await dispatch(rows, one, config.max_concurrency, progress)
        stage = (
            "awaiting_donor_review"
            if len(results) == 91 and all(r.get("stage") == "complete" for r in results)
            else "blocked"
        )
        write_json(output / "results.json", results)
        write_json(
            output / "status.json",
            {
                "stage": stage,
                "finished": len(results),
                "total": 91,
                "passed": sum(r.get("reward") == 1 for r in results),
            },
        )
        write_json(
            output / "process.json",
            {"pid": os.getpid(), "identity_sha256": identity_sha, "stage": stage},
        )


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(run))
