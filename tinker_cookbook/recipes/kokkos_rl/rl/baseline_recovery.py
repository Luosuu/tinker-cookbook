"""Run an explicitly authorized, fixed subset of failed baseline slots once.

Original results remain immutable. This produces recovery evidence only; it
does not accept a baseline, collect training data, or start training.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, TaskResult, evaluate_task
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import (
    current_task_digest,
    pinned_file,
)
from tinker_cookbook.recipes.kokkos_rl.rl.qualified_self_train import (
    qualified_evidence,
    qualified_factory,
)
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import write_json
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.sandbox.contree_polling import OperationPollPolicy, create_polling_client
from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory


@chz.chz
class Config:
    original_dir: str
    output_dir: str
    proposal_path: str
    proposal_sha256: str
    authorization_path: str
    authorization_sha256: str
    dispatch: bool = False
    concurrency: int = 2
    env_file: str = ".env"


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def checked_document(path: str, sha256: str) -> dict[str, object]:
    if pinned_file(Path(path))["sha256"] != sha256:
        raise ValueError("Recovery input changed: " + path)
    return read_json(Path(path))


def rows_by_pair(path: Path) -> dict[tuple[str, int], TaskResult]:
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        row = TaskResult(**json.loads(line))
        pair = row.task_name, row.sample_index
        if pair in result:
            raise ValueError("Duplicate result slot")
        result[pair] = row
    return result


def select_slots(
    proposal: dict[str, object],
    manifest: dict[str, object],
    original: dict[tuple[str, int], TaskResult],
) -> list[tuple[str, int]]:
    heldout = manifest.get("heldout")
    slots = proposal.get("slots")
    if not isinstance(heldout, list) or not isinstance(slots, list) or not slots:
        raise ValueError("Explicit heldout slots are required")
    expected = {(str(task), i) for task in heldout for i in range(4)}
    if len(heldout) != 20 or len(expected) != 80 or set(original) != expected:
        raise ValueError("Recovery must bind the complete original 80-slot baseline")
    selected = []
    for value in slots:
        slot = mapping(value)
        task, index = slot.get("task_name"), slot.get("sample_index")
        if not isinstance(task, str) or not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Invalid recovery slot")
        pair = task, index
        if pair not in expected or pair in selected or slot.get("heldout_only") is not True:
            raise ValueError("Recovery slots must be unique original heldout samples")
        row = original[pair]
        if row.reward != slot.get("original_reward") or row.error != slot.get("original_error"):
            raise ValueError("Original outcome differs from the proposal")
        if slot.get("task_sha256") != mapping(manifest["task_hashes"]).get(task):
            raise ValueError("Recovery task identity differs")
        selected.append(pair)
    if proposal.get("new_rollouts") != len(selected):
        raise ValueError("Additional sampling budget differs")
    return selected


def pending_slots(output: Path, slots: list[tuple[str, int]]) -> list[tuple[str, int]]:
    completed = rows_by_pair(output / "results.jsonl")
    if not set(completed).issubset(slots) or any(r.error is not None for r in completed.values()):
        raise ValueError("Unexpected or failed recovery results need a separate review")
    expected_markers = {f"{task}__{index:02d}.json": (task, index) for task, index in slots}
    for marker in (output / "attempts").glob("*.json"):
        pair = expected_markers.get(marker.name)
        if pair is None or pair not in completed:
            raise ValueError("Unresolved attempt cannot be replayed")
        claim = read_json(marker)
        if (
            (claim.get("task_name"), claim.get("sample_index")) != pair
            or claim.get("identity") != pinned_file(output / "identity.json")
            or claim.get("retry_permitted") is not False
        ):
            raise ValueError("Attempt identity changed")
    for task, index in completed:
        if not (output / "attempts" / f"{task}__{index:02d}.json").is_file():
            raise ValueError("Completed result has no exclusive attempt")
        terminal = output / "terminal" / f"{task}__{index:02d}.json"
        if not terminal.is_file() or read_json(terminal) != asdict(completed[task, index]):
            raise ValueError("Recovery result differs from its terminal evidence")
    for task, index in slots:
        if (task, index) not in completed and (
            (output / "rollouts" / f"{task}__{index:02d}").exists()
            or (output / "terminal" / f"{task}__{index:02d}.json").exists()
        ):
            raise ValueError("Partial recovery artifacts cannot be replayed")
    return [pair for pair in slots if pair not in completed]


async def dispatch_slots(
    output: Path,
    slots: list[tuple[str, int]],
    operation: Callable[[tuple[str, int]], Awaitable[TaskResult]],
    concurrency: int,
) -> list[TaskResult | BaseException | None]:
    semaphore, stop = asyncio.Semaphore(concurrency), asyncio.Event()

    async def run(pair: tuple[str, int]) -> TaskResult | None:
        async with semaphore:
            if stop.is_set():
                return None
            task, index = pair
            try:
                exclusive_json(
                    output / "attempts" / f"{task}__{index:02d}.json",
                    {
                        "task_name": task,
                        "sample_index": index,
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                        "identity": pinned_file(output / "identity.json"),
                        "retry_permitted": False,
                    },
                )
                result = await operation(pair)
                if result.error is not None:
                    stop.set()
                write_json(output / "terminal" / f"{task}__{index:02d}.json", asdict(result))
                return result
            except BaseException:
                stop.set()
                raise

    return await asyncio.gather(*(run(pair) for pair in slots), return_exceptions=True)


def validate_budget(proposal: dict[str, object], cfg: EvalConfig, count: int) -> None:
    protocol = mapping(proposal["protocol"])
    expected = {
        "max_turns": cfg.max_turns,
        "max_tokens": cfg.max_tokens,
        "max_sampled_tokens": cfg.max_sampled_tokens,
        "max_trajectory_tokens": cfg.max_trajectory_tokens,
        "max_tool_calls": cfg.max_tool_calls,
        "temperature": cfg.temperature,
        "thinking_effort": cfg.thinking_effort,
        "command_timeout": cfg.command_timeout,
        "grader_timeout": cfg.grader_timeout,
        "max_infra_retries": cfg.max_infra_retries,
        "max_concurrency": cfg.max_concurrency,
        "sandbox_build_parallelism": cfg.sandbox_build_parallelism,
        "allow_network": cfg.allow_network,
    }
    if (
        any(protocol.get(k) != v for k, v in expected.items())
        or proposal.get("model_name") != cfg.model_name
        or proposal.get("maximum_added_rollout_turns") != count * cfg.max_turns
        or proposal.get("maximum_added_sampled_tokens") != count * cfg.max_sampled_tokens
    ):
        raise ValueError("Recovery configuration exceeds or differs from the authorized protocol")


def eval_config(
    output: Path, model: str, qualified: dict[str, object], concurrency: int
) -> EvalConfig:
    return EvalConfig(
        model_name=model,
        output_path=str(output),
        resume_dir=str(output),
        max_turns=40,
        max_tokens=16384,
        temperature=1.0,
        thinking_effort=0.9,
        command_timeout=900,
        grader_timeout=900,
        max_tool_calls=80,
        max_trajectory_tokens=114688,
        max_sampled_tokens=65536,
        max_concurrency=concurrency,
        num_samples=4,
        pass_at_k="1",
        sandbox_backend="contree",
        sandbox_build_parallelism=1,
        sandbox_resource_policy=json.dumps(qualified, sort_keys=True),
        max_infra_retries=0,
        allow_network=False,
        export_kokkos_rollouts=True,
    )


async def main(config: Config) -> None:
    if not 1 <= config.concurrency <= 2:
        raise ValueError("Recovery is limited to two shared sandbox slots")
    original_dir, output = Path(config.original_dir).resolve(), Path(config.output_dir).resolve()
    if output == original_dir or original_dir in output.parents or output in original_dir.parents:
        raise ValueError("Use a separate recovery directory")
    proposal = checked_document(config.proposal_path, config.proposal_sha256)
    authorization = checked_document(config.authorization_path, config.authorization_sha256)
    if (
        authorization.get("status") != "accepted"
        or authorization.get("proposal_sha256") != config.proposal_sha256
        or authorization.get("additional_rollouts") != proposal.get("new_rollouts")
        or authorization.get("retry_on_error") is not False
    ):
        raise ValueError("Explicit matching additional-sampling authorization is required")
    sources = proposal.get("original_sources")
    if not isinstance(sources, list):
        raise ValueError("Original source proofs are required")
    for source in sources:
        proof = mapping(source)
        if pinned_file(Path(str(proof["path"])))["sha256"] != proof["sha256"]:
            raise ValueError("Original source changed")
    required_sources = {
        str(original_dir / name)
        for name in ("manifest.json", "config.json", "baseline/results.jsonl")
    }
    if not required_sources.issubset({str(mapping(p)["path"]) for p in sources}):
        raise ValueError("Original manifest, config and results must all be pinned")
    manifest = read_json(original_dir / "manifest.json")
    original = rows_by_pair(original_dir / "baseline/results.jsonl")
    slots = select_slots(proposal, manifest, original)
    original_config = read_json(original_dir / "config.json")
    tasks = load_harbor_tasks_from_dir(original_dir / "tasks")
    if {t.task_name: current_task_digest(t) for t in tasks} != manifest["task_hashes"]:
        raise ValueError("Current tasks differ from the original baseline")
    qualified = qualified_evidence(tasks, Path(str(original_config["qualified_bundle_dir"])))
    cfg = eval_config(output, str(original_config["model_name"]), qualified, config.concurrency)
    validate_budget(proposal, cfg, len(slots))
    source_paths = [Path(__file__).resolve()]
    source_paths.extend(Path(str(name)) for name in mapping(manifest["execution_sources"]))
    source_paths.append(Path(__file__).with_name("candidate_artifact.py").resolve())
    recipe_root = Path(__file__).resolve().parents[2]
    source_paths.extend(
        recipe_root / name
        for name in [
            "harbor_rl/harbor_tools.py",
            "harbor_rl/harbor_env.py",
            "kokkos_rl/rl/resource_sandbox.py",
            "kokkos_rl/rl/validate_snapshot.py",
        ]
    )
    cookbook = Path(__file__).resolve().parents[3]
    source_paths.extend(
        cookbook / name
        for name in [
            "sandbox/contree_polling.py",
            "sandbox/contree_sandbox.py",
            "completers.py",
        ]
    )
    identity = {
        "proposal": pinned_file(Path(config.proposal_path)),
        "authorization": pinned_file(Path(config.authorization_path)),
        "original_manifest": pinned_file(original_dir / "manifest.json"),
        "slots": [list(pair) for pair in slots],
        "config": chz.asdict(cfg),
        "execution_sources": [pinned_file(p) for p in sorted(set(source_paths))],
        "original_results_modified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("Recovery identity changed")
        else:
            exclusive_json(identity_path, identity)
            for proof in identity["execution_sources"]:
                source = mapping(proof)
                dest = output / "execution_sources" / str(source["sha256"])
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(Path(str(source["path"])).read_bytes())
        pending = pending_slots(output, slots)
        write_json(
            output / "preflight.json",
            {
                "pending": pending,
                "original_reused_slots": len(original) - len(slots),
                "new_model_requests": 0,
                "max_turns": 40,
                "maximum_additional_samples": len(slots),
                "dispatch": config.dispatch,
                "identity": pinned_file(identity_path),
            },
        )
        if not config.dispatch or not pending:
            return
        load_env_file(Path(config.env_file))
        cache = output / "contree_images.json"
        original_cache = Path(str(original_config["cache_path"]))
        if not cache.exists() and original_cache.is_file():
            shutil.copyfile(original_cache, cache)
        primary = ContreeDockerfileSandboxFactory(
            cache,
            timeout=3600,
            runtime_build_parallelism=1,
            allow_network=False,
        )
        primary._client = create_polling_client(primary._client.config, OperationPollPolicy())
        factory = qualified_factory(primary, tasks, str(qualified["resource_policy_version"]))
        service = tinker.ServiceClient()
        sampler = await service.create_sampling_client_async(base_model=cfg.model_name)
        tokenizer = tokenizer_utils.get_tokenizer(cfg.model_name)
        renderer = get_renderer(model_info.get_recommended_renderer_name(cfg.model_name), tokenizer)
        policy = TinkerTokenCompleter(
            sampler, max_tokens=cfg.max_tokens, temperature=cfg.temperature
        )
        by_name = {task.task_name: task for task in tasks}
        result_lock = asyncio.Lock()
        write_json(
            output / "process.json",
            {
                "pid": os.getpid(),
                "started_at": datetime.now(UTC).isoformat(),
                "identity": pinned_file(identity_path),
            },
        )
        write_json(output / "status.json", {"stage": "running", "pending": pending})

        async def run(pair: tuple[str, int]) -> TaskResult:
            task, index = pair
            for proof in identity["execution_sources"]:
                source = mapping(proof)
                if pinned_file(Path(str(source["path"]))) != source:
                    raise ValueError("Execution source changed before dispatch")
            return await evaluate_task(
                by_name[task],
                policy,
                renderer,
                factory,
                cfg,
                output,
                result_lock,
                tokenizer,
                index,
            )

        results = await dispatch_slots(output, pending, run, config.concurrency)
        failures = [str(r) for r in results if isinstance(r, BaseException)]
        terminal = rows_by_pair(output / "results.jsonl")
        write_json(
            output / "status.json",
            {
                "stage": "awaiting_recovery_review"
                if len(terminal) == len(slots)
                and all(r.error is None for r in terminal.values())
                and not failures
                else "blocked",
                "terminal": len(terminal),
                "expected": len(slots),
                "errors": sum(r.error is not None for r in terminal.values()),
                "controller_errors": failures,
                "original_results_modified": False,
                "training_started": False,
            },
        )


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
