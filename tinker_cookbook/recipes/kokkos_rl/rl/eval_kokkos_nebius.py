"""Resumable, globally bounded Nebius pass@1 sweep over verifier-gated tasks."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import chz
from openai import AsyncOpenAI

from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest, prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask, load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import (
    CLIConfig,
    OpenAITaskResult,
    evaluate_task,
    load_env_file,
)
from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory
from tinker_cookbook.utils.ml_log import dump_config

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
MODELS = (
    "zai-org/GLM-5.3-Flash",
    "moonshotai/Kimi-K3",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
)


@chz.chz
class SweepConfig:
    tasks_dir: str = "notes/experiments/kokkos-rft40/main-20260906/tasks"
    validation_dir: str = "notes/experiments/kokkos-rft40/main-20260906/validation"
    output_path: str = "notes/experiments/nebius-kokkos-pass1-20260906"
    env_file: str = ".env"
    api_key_env: str = "NEBIUS_SANDBOX_API_KEY"
    max_concurrency: int = 4
    smoke_task: str = "kokkos__kokkos-6375"
    expected_tasks: int = 100
    poll_seconds: int = 30
    modal_task_names: tuple[str, ...] = ("kokkos__kokkos-7244", "kokkos__kokkos-8164")
    modal_memory_mb: int = 16384
    modal_cpus: float = 4.0
    source_manifest: str = "notes/experiments/kokkos-rft40/main-20260906/manifest.json"
    contree_cache_path: str = "notes/experiments/SWE-kokkos-bench/v2-pass-at-k/contree_images.json"


class ImportedCacheFactory(ContreeDockerfileSandboxFactory):
    """Read a peer's cache without ever writing into that process's cache file."""

    def import_cache(self, source: Path) -> None:
        if source.is_file():
            entries = json.loads(source.read_text())
            if not isinstance(entries, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in entries.items()
            ):
                raise ValueError("Invalid source image cache")
            for key, value in entries.items():
                self._prepared_images.setdefault(key, value)


def model_config(model: str, metadata: dict[str, object], config: SweepConfig) -> CLIConfig:
    pricing = metadata["pricing"]
    if not isinstance(pricing, dict):
        raise ValueError("Model pricing is missing")
    input_price, output_price = float(pricing["prompt"]) * 1e6, float(pricing["completion"]) * 1e6
    if input_price <= 0 or output_price <= 0:
        raise ValueError("Model token prices must be positive")
    return CLIConfig(
        model_name=model,
        api_mode="chat",
        chat_provider="nebius",
        base_url=NEBIUS_BASE_URL,
        api_key_env=config.api_key_env,
        reasoning_effort="high",
        temperature=None,
        max_turns=40,
        max_tokens=16384,
        max_sampled_tokens=65536,
        max_input_tokens=5_000_000,
        max_tool_calls=80,
        grader_timeout=900,
        sandbox_build_parallelism=1,
        sandbox_resource_policy=json.dumps(
            {
                "modal_tasks": config.modal_task_names,
                "memory_mb": config.modal_memory_mb,
                "cpus": config.modal_cpus,
            },
            sort_keys=True,
        ),
        max_cost_usd_per_task=None,
        estimate_cost=True,
        input_price_per_million=input_price,
        cached_input_price_per_million=input_price,
        cache_write_price_per_million=input_price,
        output_price_per_million=output_price,
        max_infra_retries=0,
        max_concurrency=config.max_concurrency,
        tasks_dir=config.tasks_dir,
        contree_cache_path=config.contree_cache_path,
    )


def validated(task: HarborTask, validation_dir: Path) -> bool:
    path = validation_dir / f"{task.task_name}.json"
    return path.is_file() and json.loads(path.read_text()).get("passed") is True


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def summarize(results: list[OpenAITaskResult], total: int) -> dict[str, object]:
    valid = [r for r in results if r.error is None]
    passed = sum(r.reward > 0 for r in valid)
    return {
        "total_tasks": total,
        "finished": len(results),
        "valid": len(valid),
        "errors": len(results) - len(valid),
        "passed": passed,
        "pass_at_1": passed / total if len(valid) == total else None,
        "partial_pass_rate": passed / len(valid) if valid else None,
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "cached_input_tokens": sum(r.cached_input_tokens for r in results),
        "estimated_cost_usd": round(sum(r.estimated_cost_usd or 0 for r in results), 6),
        "cost_basis": "catalog prices; cached input charged at undiscounted price; not invoice",
        "mean_turns": sum(r.turns_used for r in valid) / len(valid) if valid else None,
    }


async def main(config: SweepConfig) -> None:
    root = Path(config.output_path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as controller_lock:
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A sweep controller already owns this output directory") from error
        await run_sweep(config)


async def run_sweep(config: SweepConfig) -> None:
    if config.poll_seconds < 1 or config.poll_seconds > 60:
        raise ValueError("poll_seconds must be between 1 and 60")
    if not 1 <= config.max_concurrency <= 4:
        raise ValueError("This shared experiment permits 1–4 new concurrent sandboxes")
    load_env_file(Path(config.env_file))
    key = os.environ.get(config.api_key_env)
    if not key:
        raise ValueError(f"Missing {config.api_key_env}")
    root = Path(config.output_path)
    root.mkdir(parents=True, exist_ok=True)
    tasks = load_harbor_tasks_from_dir(Path(config.tasks_dir))
    if len(tasks) != config.expected_tasks or config.smoke_task not in {t.task_name for t in tasks}:
        raise ValueError("Task snapshot cardinality or smoke task differs")
    manifest_path = Path(config.source_manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest["task_hashes"] != {t.task_name: _task_digest(t) for t in tasks}:
        raise ValueError("Task snapshot differs from the verifier manifest")
    catalog = json.loads((root / "models.json").read_text())
    metadata = {m["id"]: m for m in catalog["data"]}
    if any(m not in metadata for m in MODELS):
        raise ValueError("A requested model is unavailable; never substitute a model")
    configs = {m: model_config(m, metadata[m], config) for m in MODELS}
    model_dirs = {m: root / m.split("/")[-1] for m in MODELS}
    for model, directory in model_dirs.items():
        prepare_eval_state(
            directory, dump_config(configs[model]), tasks, evaluator="nebius-kokkos-pass1"
        )
    write_json(
        root / "launch.json",
        {
            "config": dump_config(config),
            "source_manifest": manifest,
            "pid": os.getpid(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "started_at": datetime.now(UTC).isoformat(),
        },
    )
    factory = ImportedCacheFactory(
        root / "contree_images.json",
        timeout=3600,
        runtime_build_parallelism=1,
        allow_network=False,
    )
    from tinker_cookbook.recipes.kokkos_rl.rl.resource_sandbox import (
        create_resource_sandbox_factory,
    )

    resource_factory = create_resource_sandbox_factory(
        factory,
        config.modal_task_names,
        memory_mb=config.modal_memory_mb,
        cpu=config.modal_cpus,
    )

    def gate_passed(task: HarborTask) -> bool:
        if task.task_name not in config.modal_task_names:
            return validated(task, validation_dir)
        path = root / "validation_overrides" / f"{task.task_name}.json"
        if not path.is_file():
            return False
        record = json.loads(path.read_text())
        return (
            record.get("passed") is True
            and record.get("backend") == "modal"
            and record.get("memory_mb") == config.modal_memory_mb
            and record.get("cpu") == config.modal_cpus
            and record.get("task_hash") == manifest["task_hashes"][task.task_name]
        )

    lock = asyncio.Lock()
    done: dict[tuple[str, str], OpenAITaskResult] = {}
    interrupted: set[tuple[str, str]] = set()
    for model, directory in model_dirs.items():
        for path in directory.glob("*/results.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if rows:
                result = OpenAITaskResult(**rows[-1])
                done[(model, result.task_name)] = result
        for marker in directory.glob("*/attempt_started.json"):
            pair = (model, marker.parent.name)
            if pair not in done:
                interrupted.add(pair)
    active: dict[asyncio.Task[OpenAITaskResult], tuple[str, str]] = {}
    validation_dir = Path(config.validation_dir)
    # Each model-task is sampled exactly once. Errors remain explicit and require
    # diagnosis; restarting the controller never resamples a failed trajectory.
    async with AsyncOpenAI(api_key=key, base_url=NEBIUS_BASE_URL, timeout=None) as client:
        while True:
            factory.import_cache(Path(config.contree_cache_path))
            for future in list(active):
                if future.done():
                    pair = active.pop(future)
                    done[pair] = future.result()
            smoke_done = all((m, config.smoke_task) in done for m in MODELS)
            smoke_valid = smoke_done and all(
                done[(m, config.smoke_task)].error is None for m in MODELS
            )
            queued = []
            for task in sorted(
                tasks, key=lambda t: (t.task_name != config.smoke_task, t.task_name)
            ):
                if not gate_passed(task):
                    continue
                if task.task_name != config.smoke_task and not smoke_valid:
                    continue
                for model in MODELS:
                    pair = (model, task.task_name)
                    if pair not in done and pair not in active.values() and pair not in interrupted:
                        queued.append((model, task))
            for model, task in queued[: max(0, config.max_concurrency - len(active))]:
                directory = model_dirs[model] / task.task_name
                directory.mkdir(parents=True, exist_ok=True)
                write_json(
                    directory / "attempt_started.json",
                    {
                        "model": model,
                        "task": task.task_name,
                        "started_at": datetime.now(UTC).isoformat(),
                    },
                )
                future = asyncio.create_task(
                    evaluate_task(
                        task,
                        client,
                        resource_factory,
                        configs[model],
                        directory,
                        lock,
                    )
                )
                active[future] = (model, task.task_name)
            summaries = {
                m: summarize([r for (model, _), r in done.items() if model == m], len(tasks))
                for m in MODELS
            }
            state = "running" if active else "waiting_for_verifier_gate"
            if len(done) == len(tasks) * len(MODELS):
                state = (
                    "complete"
                    if all(r.error is None for r in done.values())
                    else "needs_error_review"
                )
            elif not active and not queued and interrupted:
                state = "interrupted_needs_review"
            elif smoke_done and not smoke_valid:
                state = "smoke_error"
            write_json(
                root / "status.json",
                {
                    "state": state,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "active": list(active.values()),
                    "interrupted": sorted(interrupted),
                    "gate_passed": sum(gate_passed(t) for t in tasks),
                    "models": summaries,
                },
            )
            for model, summary in summaries.items():
                write_json(model_dirs[model] / "result.json", summary)
            if state in {
                "complete",
                "needs_error_review",
                "smoke_error",
                "interrupted_needs_review",
            }:
                break
            if active:
                await asyncio.wait(
                    active, timeout=config.poll_seconds, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                await asyncio.sleep(config.poll_seconds)


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(SweepConfig)))
