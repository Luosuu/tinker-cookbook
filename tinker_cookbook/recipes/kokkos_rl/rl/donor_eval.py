"""Once-only heldout evaluation of independently reviewed donor SFT checkpoints."""

from __future__ import annotations

import asyncio
import fcntl
import importlib.metadata
import json
import os
import shutil
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.harbor_rl.eval import TaskResult, evaluate_task
from tinker_cookbook.recipes.harbor_rl.eval_state import prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.rl.baseline_recovery import (
    dispatch_slots,
    eval_config,
    pending_slots,
    rows_by_pair,
)
from tinker_cookbook.recipes.kokkos_rl.rl.donor_qualification import records
from tinker_cookbook.recipes.kokkos_rl.rl.donor_sft import ARMS, training_evidence
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import (
    current_task_digest,
    pinned_file,
)
from tinker_cookbook.recipes.kokkos_rl.rl.qualified_self_train import (
    qualified_evidence,
    qualified_factory,
)
from tinker_cookbook.recipes.kokkos_rl.rl.saved_candidate_regrade import (
    ProofArchive,
    digest,
    exclusive_json,
)
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import delivery_metrics, write_json
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.sandbox.contree_polling import OperationPollPolicy, create_polling_client
from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory

PLAN_SHA = "955d83734b676d168840a8caa9243ae52f56a3260b0cffdbb26493cec5e9a673"
TRAINING_SHA = "d9259225ab14ca4876ebd1e6ff9b7cd3981e673be962605f801d6f263d58f74a"
MODEL = "thinkingmachines/Inkling-Small:peft:262144"


@chz.chz
class Config:
    collection_dir: str
    training_dir: str
    plan_path: str
    output_dir: str
    dispatch: bool = False
    concurrency: int = 4
    review_receipt: str | None = None
    env_file: str = ".env"


def execution_sources() -> list[dict[str, str]]:
    cookbook = Path(__file__).resolve().parents[3]
    paths = [
        "recipes/kokkos_rl/rl/donor_eval.py",
        "recipes/kokkos_rl/rl/baseline_recovery.py",
        "recipes/kokkos_rl/rl/donor_sft.py",
        "recipes/kokkos_rl/rl/qualified_self_train.py",
        "recipes/kokkos_rl/rl/rollout_data.py",
        "recipes/kokkos_rl/rl/candidate_artifact.py",
        "recipes/kokkos_rl/rl/self_train.py",
        "recipes/kokkos_rl/rl/saved_candidate_regrade.py",
        "recipes/kokkos_rl/rl/validation_recovery.py",
        "recipes/kokkos_rl/rl/nebius_phase_ledger.py",
        "recipes/kokkos_rl/rl/resource_sandbox.py",
        "recipes/kokkos_rl/rl/validate_snapshot.py",
        "recipes/harbor_rl/eval.py",
        "recipes/harbor_rl/eval_state.py",
        "recipes/harbor_rl/harbor_env.py",
        "recipes/harbor_rl/harbor_tools.py",
        "sandbox/contree_polling.py",
        "sandbox/contree_sandbox.py",
        "sandbox/modal_sandbox.py",
        "completers.py",
        "model_info.py",
        "tokenizer_utils.py",
        "rl/rollouts.py",
    ]
    return [pinned_file(cookbook / path) for path in paths]


def check_proofs(proofs: list[dict[str, str]]) -> None:
    for proof in proofs:
        if pinned_file(Path(proof["path"])) != proof:
            raise ValueError("Pinned execution or input evidence changed: " + proof["path"])


CAPACITY_POLICY = {
    "maximum_evaluation_sandboxes": 4,
    "reserved_legacy_unknown_sandboxes": 4,
    "maximum_total_sandboxes": 8,
    "paused_processes": {
        "6896": "candidate_capture_transition/resume.py",
        "90888": "provider_504_continuation_v2/run.py",
    },
}


def capacity_snapshot() -> dict[str, object]:
    observations = []
    for pid, fragment in CAPACITY_POLICY["paused_processes"].items():
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "stat=,command="], capture_output=True, text=True, check=False
        )
        if result.returncode not in (0, 1):
            raise ValueError("Cannot determine reserved process state")
        line = result.stdout.strip()
        if line:
            status, command = line.split(maxsplit=1)
            if fragment not in command or not status.startswith("T"):
                raise ValueError("Reserved Nebius process changed identity or is no longer paused")
            observations.append({"pid": int(pid), "status": status, "command": command})
        else:
            observations.append(
                {"pid": int(pid), "exited": True, "unknown_remote_reservation_retained": True}
            )
    running = subprocess.check_output(["ps", "-axo", "pid=,stat=,command="], text=True)
    for line in running.splitlines():
        if "-m tinker_cookbook.recipes.kokkos_rl.rl.donor_eval " in line:
            parts = line.strip().split(maxsplit=2)
            if int(parts[0]) != os.getpid():
                raise ValueError("Another donor evaluation controller occupies shared capacity")
    return {"policy": CAPACITY_POLICY, "processes": observations}


def validate_budget(plan: dict[str, object], cfg: object) -> None:
    expected = {
        "arms": 2,
        "heldout_tasks_per_arm": 20,
        "samples_per_task": 4,
        "total_rollouts": 160,
        "max_turns": 40,
        "max_tokens": 16384,
        "max_sampled_tokens": 65536,
        "max_trajectory_tokens": 114688,
        "max_tool_calls": 80,
        "temperature": 1.0,
        "thinking_effort": 0.9,
        "command_timeout": 900,
        "grader_timeout": 900,
        "max_infra_retries": 0,
        "shared_sandbox_concurrency": 4,
    }
    if plan.get("budget") != expected:
        raise ValueError("Original evaluation budget changed")
    actual = chz.asdict(cfg)
    for key in (
        "max_turns",
        "max_tokens",
        "max_sampled_tokens",
        "max_trajectory_tokens",
        "max_tool_calls",
        "temperature",
        "thinking_effort",
        "command_timeout",
        "grader_timeout",
        "max_infra_retries",
    ):
        if actual.get(key) != expected[key]:
            raise ValueError("Evaluator defaults changed the fixed protocol")
    if (
        actual.get("model_name") != MODEL
        or actual.get("num_samples") != 4
        or actual.get("allow_network") is not False
        or actual.get("sandbox_backend") != "contree"
        or actual.get("sandbox_build_parallelism") != 1
        or not 1 <= actual["max_concurrency"] <= 4
    ):
        raise ValueError("Evaluator runtime budget differs")


def expected_slots(manifest: dict[str, object], plan: dict[str, object]) -> list[dict[str, object]]:
    heldout, train = manifest.get("heldout"), manifest.get("train")
    hashes = mapping(manifest["task_hashes"])
    if (
        not isinstance(heldout, list)
        or not isinstance(train, list)
        or len(set(heldout)) != 20
        or len(set(train)) != 80
        or set(heldout) & set(train)
        or set(heldout + train) != set(hashes)
    ):
        raise ValueError("Exact disjoint 80/20 task split required")
    slots = [
        {"arm": arm, "task_name": task, "sample_index": index, "task_sha256": hashes[task]}
        for arm in ARMS
        for index in range(4)
        for task in heldout
    ]
    if plan.get("slots") != slots:
        raise ValueError("Evaluation plan changed its exact 160 heldout slots")
    return slots


def checked_checkpoints(training: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    report_path = training / "monitor/final_snapshot_v1/report.json"
    if pinned_file(report_path)["sha256"] != TRAINING_SHA:
        raise ValueError("Unreviewed training report")
    report = read_json(report_path)
    objects = mapping(report["objects"])
    archive = ProofArchive(report_path.parent, objects)
    archive.verify()
    proofs = [pinned_file(report_path)]
    identity = read_json(training / "identity.json")
    if report.get("training_identity") != digest(identity):
        raise ValueError("Training identity mismatch")
    checkpoints = {}
    for arm in ARMS:
        result = read_json(training / arm / "result.json")
        final, final_proofs = training_evidence(training / arm / "training")
        if (
            result.get("stage") != "complete"
            or result.get("error") is not None
            or result.get("checkpoint") != final
            or mapping(mapping(report["arms"])[arm]).get("checkpoint") != final
        ):
            raise ValueError("Incomplete or changed SFT checkpoint")
        checkpoints[arm] = str(final["sampler_path"])
        proofs.extend(final_proofs)
        proofs.append(pinned_file(training / arm / "result.json"))
    if len(set(checkpoints.values())) != 2:
        raise ValueError("Two independent final checkpoints required")
    for item in records(report["proofs"]):
        proof = mapping(item)
        archive.resolve(proof)
        proofs.append({"path": str(proof["path"]), "sha256": str(proof["sha256"])})
    review_path = training / "monitor/root_checkpoint_review.json"
    review = read_json(review_path)
    if (
        review.get("status") != "accepted_checkpoints_for_fixed_heldout_evaluation"
        or review.get("training_identity") != report["training_identity"]
        or review.get("training_report") != pinned_file(report_path)
        or review.get("evaluation_budget")
        != {
            "heldout_tasks": 20,
            "samples_per_task_per_arm": 4,
            "arms": 2,
            "total_rollouts": 160,
            "max_turns": 40,
            "max_concurrency": 4,
        }
        or any(
            mapping(mapping(review["checkpoints"])[arm]).get("sampler_path") != checkpoints[arm]
            for arm in ARMS
        )
    ):
        raise ValueError("Independent checkpoint review differs")
    proofs.append(pinned_file(review_path))
    check_proofs(proofs)
    return checkpoints, proofs


def review_identity(receipt: Path | None, identity: dict[str, object]) -> None:
    if receipt is None or read_json(receipt) != {
        "status": "accepted_checkpoint_evaluation",
        "identity_sha256": digest(identity),
        "maximum_rollouts": 160,
        "retry_permitted": False,
    }:
        raise ValueError("Exact checkpoint evaluation review required")


def check_prior_state(output: Path, all_pairs: list[tuple[str, int]]) -> None:
    state_path = output / "status.json"
    if state_path.is_file() and read_json(state_path).get("stage") == "blocked":
        raise ValueError(
            "Blocked evaluation requires explicit recovery; ordinary restart forbidden"
        )
    expected = {f"{task}__{index:02d}" for task, index in all_pairs}
    for arm in ARMS:
        for category in ("attempts", "terminal", "rollouts"):
            for path in (output / arm / category).glob("*"):
                name = path.name if category == "rollouts" else path.stem
                if name not in expected:
                    raise ValueError("Unknown slot artifact outside fixed heldout budget")


def check_candidate_or_model_stop(folder: Path, pair: tuple[str, int], result: TaskResult) -> None:
    if result.error is not None:
        return
    rollout = folder / "rollouts" / f"{pair[0]}__{pair[1]:02d}"
    candidate_path = rollout / "candidate.json"
    if candidate_path.is_file() and read_json(candidate_path).get("complete") is True:
        return
    trajectory_path = rollout / "trajectory.json"
    row = read_json(trajectory_path) if trajectory_path.is_file() else {}
    if (
        not candidate_path.exists()
        and row.get("stop_reason") == "parse_error"
        and row.get("task_name") == pair[0]
        and row.get("sample_index") == pair[1]
        and row.get("reward") == 0
        and result.reward == 0
        and row.get("turns") == result.turns_used
        and (rollout / "messages.json").is_file()
        and any((rollout / "sampling").glob("*.response.json"))
    ):
        return
    raise ValueError(
        "Candidate capture missing or incomplete; preserve original score and stop dispatch"
    )


async def run(config: Config) -> None:
    if not 1 <= config.concurrency <= 4:
        raise ValueError("Evaluation stage or shared concurrency invalid")
    collection, training, output = (
        Path(p).resolve() for p in (config.collection_dir, config.training_dir, config.output_dir)
    )
    if any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in (collection, training)
    ):
        raise ValueError("Evaluation requires an independent directory")
    plan_path = Path(config.plan_path).resolve()
    if pinned_file(plan_path)["sha256"] != PLAN_SHA:
        raise ValueError("Original fixed evaluation plan changed")
    plan = read_json(plan_path)
    manifest = read_json(collection / "manifest.json")
    if any(
        pinned_file(collection / "manifest.json")[key] != mapping(plan["original_split"])[key]
        for key in ("path", "sha256")
    ):
        raise ValueError("Original split changed")
    slots = expected_slots(manifest, plan)
    manifest_heldout = manifest["heldout"]
    assert isinstance(manifest_heldout, list)
    tasks = load_harbor_tasks_from_dir(collection / "tasks")
    if {t.task_name: current_task_digest(t) for t in tasks} != manifest["task_hashes"]:
        raise ValueError("Original frozen task payload changed")
    original_config = read_json(collection / "config.json")
    qualified = qualified_evidence(tasks, Path(str(original_config["qualified_bundle_dir"])))
    checkpoints, training_proofs = checked_checkpoints(training)
    configs = {}
    for arm in ARMS:
        cfg = eval_config(output / arm, MODEL, qualified, config.concurrency)
        cfg = chz.replace(cfg, checkpoint_url=checkpoints[arm])
        validate_budget(plan, cfg)
        configs[arm] = cfg
    identity = {
        "plan": pinned_file(plan_path),
        "training_proofs": training_proofs,
        "original_manifest": pinned_file(collection / "manifest.json"),
        "original_config": pinned_file(collection / "config.json"),
        "slots": slots,
        "checkpoints": checkpoints,
        "configs": {arm: chz.asdict(cfg) for arm, cfg in configs.items()},
        "execution_sources": execution_sources(),
        "qualified": qualified,
        "versions": {
            name: importlib.metadata.version(name) for name in ("tinker", "tml-renderers", "chz")
        },
        "maximum_rollouts": 160,
        "retry_permitted": False,
        "capacity_policy": CAPACITY_POLICY,
        "automatic_training": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("Evaluation identity changed")
        else:
            exclusive_json(identity_path, identity)
        all_pairs = [
            (str(s["task_name"]), int(str(s["sample_index"]))) for s in slots if s["arm"] == ARMS[0]
        ]
        for arm in ARMS:
            folder = output / arm
            folder.mkdir(exist_ok=True)
            if not (folder / "identity.json").exists():
                exclusive_json(
                    folder / "identity.json",
                    {
                        "phase": pinned_file(identity_path),
                        "arm": arm,
                        "checkpoint": checkpoints[arm],
                    },
                )
            elif read_json(folder / "identity.json") != {
                "phase": pinned_file(identity_path),
                "arm": arm,
                "checkpoint": checkpoints[arm],
            }:
                raise ValueError("Arm sampling identity changed")
        check_prior_state(output, all_pairs)
        pending = {arm: pending_slots(output / arm, all_pairs) for arm in ARMS}
        write_json(
            output / "preflight.json",
            {
                "identity_sha256": digest(identity),
                "pending": pending,
                "dispatch": config.dispatch,
                "maximum_rollouts": 160,
                "model_requests": 0,
                "sandbox_requests": 0,
            },
        )
        print(
            json.dumps(
                {
                    "identity_sha256": digest(identity),
                    "pending": {a: len(p) for a, p in pending.items()},
                    "dispatch": config.dispatch,
                }
            ),
            flush=True,
        )
        if not config.dispatch:
            return
        review_identity(Path(config.review_receipt) if config.review_receipt else None, identity)
        if not any(pending.values()):
            return
        write_json(output / "capacity_startup.json", capacity_snapshot())
        load_env_file(Path(config.env_file))
        cache = output / "contree_images.json"
        if not cache.exists():
            shutil.copyfile(Path(str(original_config["cache_path"])), cache)
        primary = ContreeDockerfileSandboxFactory(
            cache, timeout=3600, runtime_build_parallelism=1, allow_network=False
        )
        primary._client = create_polling_client(primary._client.config, OperationPollPolicy())
        factory = qualified_factory(primary, tasks, str(qualified["resource_policy_version"]))
        tokenizer = tokenizer_utils.get_tokenizer(MODEL)
        renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer)
        by_name = {t.task_name: t for t in tasks}
        write_json(
            output / "process.json",
            {
                "pid": os.getpid(),
                "started_at": datetime.now(UTC).isoformat(),
                "identity_sha256": digest(identity),
            },
        )
        for arm in ARMS:
            chosen = pending[arm]
            if not chosen:
                continue
            write_json(output / f"capacity_{arm}.json", capacity_snapshot())
            check_proofs(identity["execution_sources"])
            check_proofs(training_proofs)
            if (
                qualified_evidence(tasks, Path(str(original_config["qualified_bundle_dir"])))
                != qualified
            ):
                raise ValueError("Qualified execution evidence changed")
            folder, cfg = output / arm, configs[arm]
            prepare_eval_state(
                folder,
                chz.asdict(cfg),
                [by_name[str(t)] for t in manifest_heldout],
                evaluator="tinker-harbor",
            )
            write_json(folder / "config.json", chz.asdict(cfg))
            service = tinker.ServiceClient()
            sampler = await service.create_sampling_client_async(
                model_path=checkpoints[arm], base_model=MODEL
            )
            policy = TinkerTokenCompleter(
                sampler, max_tokens=cfg.max_tokens, temperature=cfg.temperature
            )
            result_lock = asyncio.Lock()
            write_json(
                output / "status.json", {"stage": "evaluating", "arm": arm, "pending": len(chosen)}
            )

            async def operation(pair: tuple[str, int]) -> TaskResult:
                check_proofs(identity["execution_sources"])
                check_proofs([identity["original_manifest"], identity["original_config"]])
                task = by_name[pair[0]]
                if current_task_digest(task) != mapping(manifest["task_hashes"])[pair[0]]:
                    raise ValueError("Task changed before dispatch")
                result = await evaluate_task(
                    task, policy, renderer, factory, cfg, folder, result_lock, tokenizer, pair[1]
                )
                check_candidate_or_model_stop(folder, pair, result)
                return result

            results = await dispatch_slots(folder, chosen, operation, config.concurrency)
            errors = [f"{type(r).__name__}: {r}" for r in results if isinstance(r, BaseException)]
            terminal = rows_by_pair(folder / "results.jsonl")
            if (
                errors
                or any(r.error is not None for r in terminal.values())
                or not set(chosen).issubset(terminal)
            ):
                write_json(
                    output / "status.json",
                    {
                        "stage": "blocked",
                        "arm": arm,
                        "errors": errors,
                        "terminal": len(terminal),
                        "automatic_restart": False,
                    },
                )
                return
            write_json(
                folder / "efficiency.json", delivery_metrics(list(terminal.values()), folder)
            )
        write_json(
            output / "status.json",
            {
                "stage": "awaiting_evaluation_review",
                "terminal": {
                    arm: len(rows_by_pair(output / arm / "results.jsonl")) for arm in ARMS
                },
                "automatic_training": False,
            },
        )


async def main(config: Config) -> None:
    try:
        await run(config)
    except BaseException as error:
        output = Path(config.output_dir).resolve()
        if config.dispatch and (output / "identity.json").is_file():
            failure = {
                "stage": "blocked",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "at": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "automatic_restart": False,
            }
            exclusive_json(
                output
                / "controller_failures"
                / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json",
                failure,
            )
            # A rejected second controller must never overwrite the live owner's status.
            with (output / "controller.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    write_json(output / "status.json", failure)
        raise


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
