"""Fill never-attempted pairs only after the original queue releases a model slot."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.eval_state import prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import (
    MODELS,
    ImportedCacheFactory,
    write_json,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import (
    CLIConfig,
    evaluate_task,
    load_env_file,
)
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import (
    attempt_artifacts,
    claim_pair,
    current_task_digest,
    eligible_evidence,
    original_gate,
    original_model_capacity,
    pinned_file,
    scope_approval,
)
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_request_audit import (
    audited_client,
    write_exclusive,
)
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_transport import (
    NetworkPolicy,
    create_nebius_client,
)
from tinker_cookbook.recipes.kokkos_rl.rl.resource_sandbox import create_resource_sandbox_factory
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import policy_for_task
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json
from tinker_cookbook.sandbox.contree_polling import OperationPollPolicy, create_polling_client
from tinker_cookbook.utils.ml_log import dump_config


@chz.chz
class Config:
    original_root: str
    output_path: str
    snapshot_dir: str
    qualification_path: str
    coverage_path: str
    scope_approval_path: str
    resource_policy_version: str = "runtime_v12"
    required_environment_policies: tuple[tuple[str, str], ...] = (
        ("kokkos__pykokkos-422", "dockerfile_env_v1"),
    )
    env_file: str = ".env"
    dispatch: bool = False
    max_new_pairs: int | None = 1
    poll_seconds: int = 30


def evaluation_config(
    saved: dict[str, object], model: str, execution_policy: dict[str, object]
) -> CLIConfig:
    required = {
        "model_name": model,
        "api_mode": "chat",
        "chat_provider": "nebius",
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
    if any(saved.get(k) != v for k, v in required.items()):
        raise ValueError("The original sampling protocol differs from the authorized experiment")
    return CLIConfig(
        **{
            **saved,
            "sandbox_build_parallelism": None,
            "sandbox_resource_policy": json.dumps(execution_policy, sort_keys=True),
            "max_concurrency": 1,
            "max_infra_retries": 0,
        }
    )


def require_complete_review(
    hashes: dict[str, str], qualification_path: Path, coverage_path: Path, approvals: Path
) -> None:
    qualification, review = read_json(qualification_path), read_json(coverage_path)
    if (
        len(hashes) != 100
        or qualification.get("task_hashes") != hashes
        or review.get("task_hashes") != hashes
        or qualification.get("ready_for_sampling") is not True
        or qualification.get("qualified") != 100
        or review.get("status") != "complete"
        or review.get("blockers") != []
    ):
        raise ValueError("A complete exact-hash verifier and positive scope review is required")
    for name, digest in hashes.items():
        scope_approval(approvals, name, digest)


def validate_catalog(catalog: dict[str, object], configs: dict[str, CLIConfig]) -> None:
    rows = catalog.get("data")
    if not isinstance(rows, list):
        raise ValueError("The model catalog is unavailable")
    entries = {str(mapping(row)["id"]): mapping(row) for row in rows}
    if not set(MODELS).issubset(entries):
        raise ValueError("An exact requested model is unavailable")
    for model in MODELS:
        if not isinstance(entries[model].get("pricing"), dict):
            raise ValueError("Verbose catalog pricing is unavailable; refuse generation")
        prices = mapping(entries[model]["pricing"])
        expected = configs[model]
        if Decimal(str(prices.get("prompt"))) * 1_000_000 != Decimal(
            str(expected.input_price_per_million)
        ) or Decimal(str(prices.get("completion"))) * 1_000_000 != Decimal(
            str(expected.output_price_per_million)
        ):
            raise ValueError(
                "Provider catalog pricing changed; review the cost identity before dispatch"
            )


async def run(config: Config) -> None:
    if config.max_new_pairs is not None and (
        isinstance(config.max_new_pairs, bool) or config.max_new_pairs < 1
    ):
        raise ValueError("The per-launch pair limit must be positive or None")
    if not 1 <= config.poll_seconds <= 60:
        raise ValueError("Polling interval must be within one to sixty seconds")
    original, phase = Path(config.original_root), Path(config.output_path)
    if phase.resolve() == original.resolve():
        raise ValueError("Use a separate output directory")
    snapshot = Path(config.snapshot_dir)
    tasks = load_harbor_tasks_from_dir(snapshot / "tasks")
    hashes = {task.task_name: current_task_digest(task) for task in tasks}
    if read_json(snapshot / "manifest.json").get("task_hashes") != hashes:
        raise ValueError("Snapshot manifest differs from actual payload")
    qualification, coverage, approvals = (
        Path(config.qualification_path),
        Path(config.coverage_path),
        Path(config.scope_approval_path),
    )
    require_complete_review(hashes, qualification, coverage, approvals)
    launch = read_json(original / "launch.json")
    old_config = mapping(launch["config"])
    launch_proof = pinned_file(original / "launch.json")
    manifest_proof = pinned_file(Path(str(old_config["source_manifest"])))
    policies = {
        task.task_name: policy_for_task(task, config.resource_policy_version) for task in tasks
    }
    environments = dict(config.required_environment_policies)
    if (
        len(environments) != len(config.required_environment_policies)
        or set(environments) - hashes.keys()
    ):
        raise ValueError("Ambiguous or unknown runtime environment task")
    execution_policy = {
        "resources": {name: asdict(policy) for name, policy in policies.items()},
        "resource_policy_version": config.resource_policy_version,
        "required_environment_policies": environments,
        "operation_poll_policy": asdict(OperationPollPolicy()),
        "network_policy": NetworkPolicy().identity(),
        "generation_audit": "exclusive_request_and_raw_response_v1",
        "shared_model_limit": 4,
        "this_phase_model_limit": 1,
    }
    configs = {}
    sources = {}
    for model in MODELS:
        path = original / model.split("/")[-1] / "eval_identity.json"
        saved = read_json(path)
        if saved.get("tasks") != mapping(launch["source_manifest"])["task_hashes"]:
            raise ValueError("Original model payload identity differs")
        configs[model] = evaluation_config(
            mapping(mapping(saved["identity"])["config"]), model, execution_policy
        )
        sources[model] = pinned_file(path)
    phase.mkdir(parents=True, exist_ok=True)
    identity_path = phase / "phase_identity.json"
    if identity_path.exists():
        saved_reserved = read_json(identity_path)["reserved_original_tasks"]
        if not isinstance(saved_reserved, list) or any(
            not isinstance(n, str) for n in saved_reserved
        ):
            raise ValueError("Original reserved queue must be an explicit list of task names")
        reserved = frozenset(str(n) for n in saved_reserved)
    else:
        reserved, _ = original_gate(original)
    identity = {
        "phase_root": str(phase.resolve()),
        "snapshot": pinned_file(snapshot / "manifest.json"),
        "task_hashes": hashes,
        "qualification": pinned_file(qualification),
        "coverage_review": pinned_file(coverage),
        "scope_approvals": pinned_file(approvals),
        "original_launch": launch_proof,
        "original_manifest": manifest_proof,
        "original_model_identities": sources,
        "reserved_original_tasks": sorted(reserved),
        "execution_policy": execution_policy,
        "model_configs": {name: dump_config(value) for name, value in configs.items()},
    }
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Phase identity changed; use another directory after review")
    if not identity_path.exists():
        write_exclusive(identity_path, identity)
    identity_sha = pinned_file(identity_path)["sha256"]
    ledger = original / "complementary_pair_ledger"
    pending = []
    completed = []
    for task in sorted(tasks, key=lambda task: task.task_name):
        if task.task_name in reserved:
            continue
        for model in MODELS:
            if attempt_artifacts(original, model, task.task_name):
                continue
            trial = phase / model.split("/")[-1] / task.task_name
            claim = ledger / model.split("/")[-1] / task.task_name / "claim.json"
            if attempt_artifacts(phase, model, task.task_name) or claim.exists():
                if (
                    not claim.exists()
                    or read_json(claim).get("phase_identity_sha256") != identity_sha
                    or read_json(claim).get("phase_root") != str(phase.resolve())
                ):
                    raise ValueError("An earlier phase owns this pair")
                result = trial / "results.jsonl"
                if not result.exists():
                    raise ValueError("An interrupted attempt requires explicit recovery review")
                rows = [
                    json.loads(line) for line in result.read_text().splitlines() if line.strip()
                ]
                if len(rows) != 1 or mapping(rows[0]).get("task_name") != task.task_name:
                    raise ValueError("Ambiguous existing result")
                if (trial / "requests").exists() and list(
                    (trial / "requests").glob("*/unreceived_or_unpersisted_response.json")
                ):
                    raise ValueError("Uncertain generation usage requires explicit recovery review")
                completed.append({"model": model, **mapping(rows[0])})
                continue
            evidence = eligible_evidence(
                task=task,
                model=model,
                original_root=original,
                reserved_original_tasks=reserved,
                qualification_path=qualification,
                coverage_path=coverage,
                scope_approval_path=approvals,
                policy=policies[task.task_name],
                required_environment_policy=environments.get(task.task_name),
            )
            pending.append((task, model, evidence))
    capacity = original_model_capacity(
        original,
        reserved,
        launch_proof=launch_proof,
        manifest_proof=manifest_proof,
        now=datetime.now(UTC),
    )
    write_json(
        phase / "preflight.json",
        {
            "dispatch_requested": config.dispatch,
            "phase_identity_sha256": identity_sha,
            "pending_pairs": [
                {"model": model, "task": task.task_name} for task, model, _ in pending
            ],
            "completed": completed,
            "capacity": capacity,
            "new_model_requests": 0,
            "new_claims": 0,
        },
    )
    if not config.dispatch:
        return
    for model, values in configs.items():
        prepare_eval_state(
            phase / model.split("/")[-1],
            dump_config(values),
            tasks,
            evaluator="nebius-complement-pass1",
        )
    launch_record = {
        "pid": os.getpid(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": datetime.now(UTC).isoformat(),
        "identity_sha256": identity_sha,
        "invocation": dump_config(config),
    }
    launch_path = phase / "launches" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json"
    write_exclusive(launch_path, launch_record)
    write_json(phase / "process.json", launch_record)
    load_env_file(Path(config.env_file))
    first = configs[MODELS[0]]
    primary = None
    dispatched = 0
    paused = False
    lock = asyncio.Lock()
    async with create_nebius_client(
        api_key=os.environ[first.api_key_env], base_url=first.base_url or ""
    ) as client:
        for task, model, evidence in pending:
            while True:
                capacity = original_model_capacity(
                    original,
                    reserved,
                    launch_proof=launch_proof,
                    manifest_proof=manifest_proof,
                    now=datetime.now(UTC),
                )
                write_json(
                    phase / "status.json",
                    {
                        "state": "waiting_for_shared_model_slot",
                        "capacity": capacity,
                        "completed": completed,
                    },
                )
                available = capacity["available_model_slots"]
                if isinstance(available, int) and not isinstance(available, bool) and available > 0:
                    break
                await asyncio.sleep(config.poll_seconds)
            if primary is None:
                # Nebius exposes prices only on its documented verbose catalog.
                catalog = await client.models.list(extra_query={"verbose": "true"})
                catalog_record = catalog.model_dump(mode="json")
                write_exclusive(
                    phase
                    / "catalog_checks"
                    / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json",
                    catalog_record,
                )
                validate_catalog(catalog_record, configs)
                primary = ImportedCacheFactory(
                    phase / "contree_images.json",
                    timeout=3600,
                    runtime_build_parallelism=1,
                    allow_network=False,
                )
                primary._client = create_polling_client(
                    primary._client.config, OperationPollPolicy()
                )
            primary.import_cache(Path(str(old_config["contree_cache_path"])))
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
            claim = claim_pair(
                task=task,
                ledger_root=ledger,
                phase_root=phase,
                original_root=original,
                reserved_original_tasks=reserved,
                evidence=evidence,
                phase_identity_sha256=identity_sha,
            )
            trial = phase / model.split("/")[-1] / task.task_name
            write_exclusive(
                trial / "attempt_started.json",
                {
                    "model": model,
                    "task": task.task_name,
                    "task_hash": hashes[task.task_name],
                    "phase_identity_sha256": identity_sha,
                    "code_commit": launch_record["commit"],
                    "launch": pinned_file(launch_path),
                    "claim": pinned_file(claim),
                    "started_at": datetime.now(UTC).isoformat(),
                },
            )
            write_json(
                phase / "status.json",
                {
                    "state": "running",
                    "active": [[model, task.task_name]],
                    "capacity_before_dispatch": capacity,
                    "completed": completed,
                },
            )
            result = await evaluate_task(
                task,
                audited_client(client, trial / "requests", identity_sha),
                factory,
                configs[model],
                trial,
                lock,
                before_grading=partial(capture_candidate, task=task, results_dir=trial),
            )
            completed.append({"model": model, **asdict(result)})
            write_json(
                phase / "status.json",
                {"state": "pair_complete", "active": [], "completed": completed},
            )
            if list((trial / "requests").glob("*/unreceived_or_unpersisted_response.json")):
                raise RuntimeError(
                    "Uncertain generation remains explicit; stop dispatch for review"
                )
            dispatched += 1
            if config.max_new_pairs is not None and dispatched >= config.max_new_pairs:
                paused = True
                break
    write_json(
        phase / "status.json",
        {
            "state": "paused_after_requested_pairs" if paused else "complete",
            "active": [],
            "completed": completed,
        },
    )


async def main(config: Config) -> None:
    ledger = Path(config.original_root) / "complementary_pair_ledger"
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await run(config)


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(main))
