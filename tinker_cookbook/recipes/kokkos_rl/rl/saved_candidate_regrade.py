"""Grade immutable saved candidates once under an explicitly reviewed verifier.

No inference client is constructed. A failed or interrupted grading attempt is
never retried automatically, and original rewards and usage remain untouched.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import inspect
import json
import os
import re
import subprocess
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    SandboxFactory,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import (
    MODELS,
    ImportedCacheFactory,
    write_json,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import (
    current_task_digest,
    original_model_capacity,
    pinned_file,
)
from tinker_cookbook.recipes.kokkos_rl.rl.qualified_self_train import qualified_evidence
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import grade_patch
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import (
    Policy,
    pid_alive,
    policy_for_task,
)
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json
from tinker_cookbook.sandbox.contree_polling import OperationPollPolicy, create_polling_client


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected an explicit list")
    return value


def exclusive_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class ProofArchive:
    """Content-addressed byte copies, retaining original paths as provenance."""

    def __init__(self, root: Path, entries: dict[str, object] | None = None):
        self.root = root.resolve()
        self.entries = entries if entries is not None else {}

    def add(self, proof: dict[str, object]) -> dict[str, str]:
        source = Path(str(proof["path"])).resolve()
        raw = source.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != proof.get("sha256"):
            raise ValueError("Source changed before freezing")
        target = self.root / "objects" / sha
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("Frozen object changed")
        else:
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        record = pinned_file(target)
        previous = self.entries.get(str(source))
        if previous is not None and previous != record:
            raise ValueError("Two revisions of one source in a frozen bundle")
        self.entries[str(source)] = record
        return record

    def resolve(self, proof: dict[str, object]) -> Path:
        frozen = mapping(self.entries.get(str(Path(str(proof["path"])).resolve())))
        path = Path(str(frozen["path"])).resolve()
        if not path.is_relative_to(self.root / "objects"):
            raise ValueError("Frozen proof escapes its archive")
        if frozen.get("sha256") != proof.get("sha256") or pinned_file(path) != frozen:
            raise ValueError("Frozen proof differs from approved source")
        return path

    def verify(self) -> None:
        for original, value in self.entries.items():
            row = mapping(value)
            self.resolve({"path": original, "sha256": row["sha256"]})


def verifier_identity(task: HarborTask) -> dict[str, object]:
    """Policy plus exact verifier/environment bytes; filenames alone are insufficient."""
    files = {
        str(p.relative_to(task.task_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((task.task_dir / "tests").rglob("*"))
        if p.is_file()
    }
    if "tests/test.sh" not in files:
        raise ValueError("Verifier script is missing")
    return {
        "tests": files,
        "environment_sha256": pinned_file(task.task_dir / "environment/Dockerfile")["sha256"],
        "base_commit": read_json(task.task_dir / "metadata.json")["base_commit"],
        "resource_policy": asdict(policy_for_task(task, "runtime_v12")),
    }


def prior_disposition(prior: dict[str, object], current: dict[str, object], patch_sha: str) -> str:
    if prior.get("patch_sha256") != patch_sha:
        raise ValueError("Prior grading used another candidate")
    if prior.get("verifier_identity") != current:
        return "new_verifier"
    if (
        prior.get("stage") == "complete"
        and prior.get("error") is None
        and prior.get("reward") in (0, 1)
    ):
        return "reuse_existing_score"
    return "blocked_prior_attempt"


def code_identity() -> dict[str, str]:
    return {
        "runner_sha256": pinned_file(Path(__file__))["sha256"],
        **{
            obj.__name__: hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
            for obj in (
                grade_patch,
                HarborReward,
                ImportedCacheFactory,
                create_polling_client,
                qualified_evidence,
                current_task_digest,
                policy_for_task,
            )
        },
    }


def reused_result(row: dict[str, object]) -> dict[str, object]:
    priors = row.get("prior_gradings")
    if not isinstance(priors, list):
        raise ValueError("Prior grading evidence missing")
    matches = [
        mapping(p)
        for p in priors
        if prior_disposition(
            mapping(p),
            mapping(row["verifier_identity"]),
            str(mapping(row["candidate"])["patch_sha256"]),
        )
        == "reuse_existing_score"
    ]
    if not matches or len({p["reward"] for p in matches}) != 1:
        raise ValueError("Existing same-verifier scores are missing or contradictory")
    return {
        "stage": "reused",
        "model": row["model"],
        "task": row["task"],
        "split": row["split"],
        "reward": matches[0]["reward"],
        "error": None,
        "prior_gradings": matches,
        "model_requests": 0,
        "added_inference_cost_usd": 0,
    }


async def drain_dispatch(
    records: list[dict[str, object]],
    worker: Callable[[dict[str, object]], Awaitable[dict[str, object]]],
    max_concurrency: int,
    on_progress: Callable[[list[dict[str, object]]], None],
) -> list[dict[str, object]]:
    """Stop new work after a guard fails; let already started grading finish."""
    semaphore = asyncio.Semaphore(max_concurrency)
    stopped = asyncio.Event()
    results: list[dict[str, object]] = []
    progress_errors: list[dict[str, object]] = []

    async def one(row: dict[str, object]) -> None:
        async with semaphore:
            if stopped.is_set():
                result = {
                    "model": row["model"],
                    "task": row["task"],
                    "stage": "blocked_not_started",
                    "model_requests": 0,
                }
            else:
                try:
                    result = await worker(row)
                except Exception as error:
                    stopped.set()
                    result = {
                        "model": row["model"],
                        "task": row["task"],
                        "stage": "guard_or_partial_failure",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                        "model_requests": 0,
                    }
            results.append(result)
            try:
                on_progress(results)
            except Exception as error:
                stopped.set()
                progress_errors.append(
                    {
                        "stage": "guard_or_partial_failure",
                        "error": f"{type(error).__name__}: {error}",
                        "operation": "persist_progress",
                    }
                )

    await asyncio.gather(*(one(row) for row in records))
    return results + progress_errors


def verify_record(
    row: dict[str, object],
    archive: ProofArchive,
    task: HarborTask,
    original: HarborTask,
    split: dict[str, object],
) -> Path:
    name, model = task.task_name, row.get("model")
    if model not in MODELS or row.get("task") != name:
        raise ValueError("Unexpected model/task pair")
    if (
        row.get("eligible_for_future_explicit_patch_only_regrade") is not True
        or row.get("blockers") != []
    ):
        raise ValueError("Candidate has not been positively approved")
    if current_task_digest(task) != row.get("final_task_hash") or current_task_digest(
        original
    ) != row.get("original_task_hash"):
        raise ValueError("Task payload changed")
    expected_split = (
        "train"
        if name in sequence(split["train"])
        else "heldout"
        if name in sequence(split["heldout"])
        else None
    )
    if expected_split is None or row.get("split") != expected_split:
        raise ValueError("Training/heldout membership changed")
    proofs = row.get("source_proofs")
    if not isinstance(proofs, list):
        raise ValueError("Source proofs missing")
    for proof in proofs:
        archive.resolve(mapping(proof))
    for filename in ("instruction.md", "environment/Dockerfile", "task.toml"):
        check = mapping(mapping(row["source_file_checks"])[filename])
        old = archive.resolve(mapping(check["original"])).read_bytes()
        new = archive.resolve(mapping(check["final"])).read_bytes()
        if (
            check.get("identical") is not True
            or old != new
            or new != (task.task_dir / filename).read_bytes()
            or old != (original.task_dir / filename).read_bytes()
        ):
            raise ValueError("Model prompt or environment differs")
    policy = asdict(policy_for_task(task, "runtime_v12"))
    if (
        policy != asdict(Policy())
        or row.get("final_resource_policy") != policy
        or row.get("original_resource_policy") != policy
    ):
        raise ValueError("This phase only permits the reviewed unchanged ConTree resource policy")
    candidate = read_json(archive.resolve(mapping(row["candidate_proof"])))
    patch = archive.resolve(mapping(row["patch_proof"]))
    baseline = read_json(task.task_dir / "metadata.json")["base_commit"]
    if read_json(original.task_dir / "metadata.json").get("base_commit") != baseline:
        raise ValueError("Original and final repository bases differ")
    if (
        candidate != row.get("candidate")
        or candidate.get("complete") is not True
        or candidate.get("stage") != "before_hidden_tests"
        or not candidate.get("sandbox_id")
    ):
        raise ValueError("Candidate execution provenance is incomplete")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(baseline))
        or candidate.get("base_commit") != baseline
        or candidate.get("head_commit") != baseline
    ):
        raise ValueError("Candidate base/HEAD differs")
    if (
        candidate.get("patch_bytes") != patch.stat().st_size
        or candidate.get("patch_sha256") != pinned_file(patch)["sha256"]
    ):
        raise ValueError("Candidate bytes changed")
    identity = read_json(archive.resolve(mapping(row["model_identity_proof"])))
    config = mapping(mapping(identity["identity"])["config"])
    required = {
        "model_name": model,
        "chat_provider": "nebius",
        "api_mode": "chat",
        "reasoning_effort": "high",
        "temperature": None,
        "max_turns": 40,
        "max_tokens": 16384,
        "max_sampled_tokens": 65536,
        "max_input_tokens": 5_000_000,
        "max_tool_calls": 80,
        "command_timeout": 900,
        "grader_timeout": 900,
        "allow_network": False,
    }
    if (
        any(config.get(k) != v for k, v in required.items())
        or mapping(identity["tasks"]).get(name) != row["original_task_hash"]
    ):
        raise ValueError("Original sampling identity differs")
    marker = read_json(archive.resolve(mapping(row["marker_proof"])))
    if marker.get("task") != name or marker.get("model") != model:
        raise ValueError("Original attempt identity differs")
    result_rows = archive.resolve(mapping(row["result_proof"])).read_text().splitlines()
    if len(result_rows) != 1:
        raise ValueError("Ambiguous original result")
    result = mapping(json.loads(result_rows[0]))
    if result.get("task_name") != name:
        raise ValueError("Original result belongs to another task")
    trace = json.loads(archive.resolve(mapping(row["trace_proof"])).read_text())
    if (
        not isinstance(trace, list)
        or len(trace) != result.get("turns_used")
        or len(trace) > 40
        or sum(len(sequence(mapping(t)["function_calls"])) for t in trace)
        != result.get("tool_calls")
    ):
        raise ValueError("Original trajectory accounting differs")
    if row.get("verifier_identity") != verifier_identity(task):
        raise ValueError("Reviewed verifier identity changed")
    priors = row.get("prior_gradings")
    if not isinstance(priors, list):
        raise ValueError("Explicit prior-grading review is required")
    dispositions = [
        prior_disposition(mapping(p), verifier_identity(task), str(candidate["patch_sha256"]))
        for p in priors
    ]
    decision = (
        "blocked_prior_attempt"
        if "blocked_prior_attempt" in dispositions
        else "reuse_existing_score"
        if "reuse_existing_score" in dispositions
        else "new_verifier"
    )
    if row.get("decision") != decision or decision == "blocked_prior_attempt":
        raise ValueError("Repeated same-verifier grading requires separate review")
    if decision == "reuse_existing_score":
        reused_result(row)
    return patch


def verify_bundle(root: Path) -> tuple[dict[str, object], dict[str, HarborTask], ProofArchive]:
    manifest = read_json(root / "manifest.json")
    archive = ProofArchive(root, mapping(manifest["archive"]))
    archive.verify()
    tasks_list = load_harbor_tasks_from_dir(root / "tasks")
    tasks = {t.task_name: t for t in tasks_list}
    old = {t.task_name: t for t in load_harbor_tasks_from_dir(root / "original_tasks")}
    evidence = qualified_evidence(tasks_list, root / "qualified")
    if (
        evidence != manifest.get("qualification")
        or manifest.get("code_identity") != code_identity()
    ):
        raise ValueError("Qualification or grading implementation changed")
    split = read_json(archive.resolve(mapping(manifest["split_proof"])))
    train, heldout = split.get("train"), split.get("heldout")
    if (
        not isinstance(train, list)
        or not isinstance(heldout, list)
        or len(train) != 80
        or len(heldout) != 20
        or set(train) & set(heldout)
        or set(train + heldout) != set(tasks)
    ):
        raise ValueError("Original 80/20 split is invalid")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 83:
        raise ValueError("Expected the reviewed 83 saved candidates")
    seen = set()
    for value in records:
        row = mapping(value)
        key = (row.get("model"), row.get("task"))
        if key in seen:
            raise ValueError("Duplicate candidate pair")
        seen.add(key)
        name = str(row["task"])
        verify_record(row, archive, tasks[name], old[name], split)
    return manifest, tasks, archive


async def grade_once(
    row: dict[str, object],
    task: HarborTask,
    factory: SandboxFactory,
    patch: Path,
    folder: Path,
    identity_sha: str,
) -> dict[str, object]:
    result_path = folder / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if (
            result.get("identity_sha256") != identity_sha
            or result.get("patch_sha256") != pinned_file(patch)["sha256"]
            or result.get("model") != row["model"]
            or result.get("task") != task.task_name
            or result.get("task_hash") != row["final_task_hash"]
        ):
            raise ValueError("Existing grading result belongs to another candidate/identity")
        return result
    if (folder / "attempt_started.json").exists():
        raise ValueError("Interrupted grading is retained and must not be retried")
    record: dict[str, object] = {
        "model": row["model"],
        "task": task.task_name,
        "split": row["split"],
        "task_hash": row["final_task_hash"],
        "patch_sha256": pinned_file(patch)["sha256"],
        "identity_sha256": identity_sha,
        "model_requests": 0,
        "added_inference_cost_usd": 0,
        "started_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    exclusive_json(folder / "attempt_started.json", record)
    try:
        reward = await grade_patch(task, factory, patch.read_text(), folder / "grading.json")
        record.update(stage="complete", reward=reward, error=None)
    except Exception as error:
        record.update(
            stage="infra_error",
            reward=None,
            error=f"{type(error).__name__}: {error}",
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
    exclusive_json(result_path, record)
    return record


@chz.chz
class Config:
    bundle_dir: str
    output_dir: str
    original_root: str
    cache_source: str
    native_pid: int = 90949
    max_concurrency: int = 2
    dispatch: bool = False
    env_file: str = ".env"


async def run(config: Config) -> None:
    if not 1 <= config.max_concurrency <= 2:
        raise ValueError("Only two sandbox slots are allocated to this phase")
    root, output = Path(config.bundle_dir).resolve(), Path(config.output_dir).resolve()
    if root == output or root in output.parents:
        raise ValueError("Mutable execution output must be outside the immutable input bundle")
    manifest, tasks, archive = verify_bundle(root)
    manifest_proof = pinned_file(root / "manifest.json")
    identity = {
        "bundle": manifest_proof,
        "grading_code": code_identity(),
        "operation_poll_policy": asdict(OperationPollPolicy()),
        "max_concurrency": config.max_concurrency,
        "sandbox_budget": {"total": 8, "other_reserved": 6, "this_phase": 2},
        "model_requests": 0,
        "added_inference_cost_usd": 0,
    }
    identity_sha = digest(identity)
    records = [mapping(x) for x in sequence(manifest["records"])]
    if any(
        (output / str(r["model"]).split("/")[-1] / str(r["task"]) / "attempt_started.json").exists()
        and not (output / str(r["model"]).split("/")[-1] / str(r["task"]) / "result.json").exists()
        for r in records
    ):
        raise ValueError("Interrupted grading requires diagnosis before any further dispatch")
    if not config.dispatch:
        print(
            json.dumps(
                {
                    "identity": identity,
                    "identity_sha256": identity_sha,
                    "records": len(records),
                    "new_verifier": sum(r["decision"] == "new_verifier" for r in records),
                    "reused": sum(r["decision"] == "reuse_existing_score" for r in records),
                    "remote_requests": 0,
                }
            )
        )
        return
    if pid_alive(config.native_pid):
        raise ValueError("Original native baseline still holds its sandbox reservation")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = output / "identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Grading phase identity changed")
        if not identity_path.exists():
            exclusive_json(identity_path, identity)
        write_json(
            output / "process.json",
            {"pid": os.getpid(), "stage": "running", "identity_sha256": identity_sha},
        )
        load_env_file(Path(config.env_file))
        primary = ImportedCacheFactory(
            output / "contree_images.json",
            timeout=3600,
            runtime_build_parallelism=1,
            allow_network=False,
        )
        primary._client = create_polling_client(primary._client.config, OperationPollPolicy())
        original_root = Path(config.original_root)
        launch = read_json(original_root / "launch.json")
        launch_proof = pinned_file(original_root / "launch.json")
        source_manifest = pinned_file(Path(str(mapping(launch["config"])["source_manifest"])))

        async def one(row: dict[str, object]) -> dict[str, object]:
            if (
                pid_alive(config.native_pid)
                or pinned_file(root / "manifest.json") != manifest_proof
            ):
                raise ValueError("Resource reservation or immutable bundle changed")
            capacity = original_model_capacity(
                original_root,
                frozenset(),
                launch_proof=launch_proof,
                manifest_proof=source_manifest,
                now=datetime.now(UTC),
            )
            if capacity["available_model_slots"] != 1:
                raise ValueError("Original three-unknown-request reservation changed")
            name = str(row["task"])
            patch = archive.resolve(mapping(row["patch_proof"]))
            if current_task_digest(tasks[name]) != row["final_task_hash"]:
                raise ValueError("Task changed before grading")
            primary.import_cache(Path(config.cache_source))
            folder = output / str(row["model"]).split("/")[-1] / name
            if row["decision"] == "reuse_existing_score":
                return reused_result(row)
            return await grade_once(row, tasks[name], primary, patch, folder, identity_sha)

        def progress(results: list[dict[str, object]], stage: str = "running") -> None:
            write_json(
                output / "status.json",
                {
                    "stage": stage,
                    "finished": sum(
                        r.get("stage") in {"complete", "infra_error", "reused"} for r in results
                    ),
                    "total": len(records),
                    "passed": sum(r.get("reward") == 1 for r in results),
                    "infra_errors": sum(r.get("stage") == "infra_error" for r in results),
                    "blocked": sum(
                        r.get("stage") in {"blocked_not_started", "guard_or_partial_failure"}
                        for r in results
                    ),
                },
            )

        results = await drain_dispatch(records, one, config.max_concurrency, progress)
        stage = (
            "blocked"
            if any(
                r.get("stage") in {"blocked_not_started", "guard_or_partial_failure", "infra_error"}
                for r in results
            )
            else "complete"
        )
        write_json(output / "results.json", results)
        progress(results, stage)
        write_json(
            output / "process.json",
            {"pid": os.getpid(), "stage": stage, "identity_sha256": identity_sha},
        )


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(run))
