"""Evaluate previously unsampled resource-exception tasks in a separate phase."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import chz
from openai import AsyncOpenAI

from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest, prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask, load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import MODELS, summarize, write_json
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import (
    CLIConfig,
    OpenAITaskResult,
    evaluate_task,
    load_env_file,
)
from tinker_cookbook.recipes.kokkos_rl.rl.resource_sandbox import create_resource_sandbox_factory
from tinker_cookbook.sandbox import SandboxInterface


@chz.chz
class ResourcePhaseConfig:
    source_root: str
    task_names: tuple[str, ...]
    validation_files: tuple[str, ...]
    max_concurrency: int = 4


def validate_resource_record(record: dict[str, object], *, task_name: str, task_hash: str) -> None:
    expected = {
        "task": task_name,
        "task_hash": task_hash,
        "passed": True,
        "nop": 0.0,
        "oracle": 1.0,
        "backend": "modal",
        "memory_mb": 16384,
        "cpu": 4.0,
        "build_parallelism": 4,
        "grader_timeout": 900,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Resource validation does not match the exact task and execution policy")


async def main(config: ResourcePhaseConfig) -> None:
    if not config.task_names or len(set(config.task_names)) != len(config.task_names):
        raise ValueError("Resource phase requires unique explicit task names")
    if not 1 <= config.max_concurrency <= 4:
        raise ValueError("The resource phase shares at most four inference slots")
    with (Path(config.source_root) / "controller.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Original controller still owns the shared inference slots"
            ) from error
        await run_phase(config)


async def run_phase(config: ResourcePhaseConfig) -> None:
    root = Path(config.source_root)
    phase = root / "resource_phase_j4"
    launch = json.loads((root / "launch.json").read_text())
    tasks = load_harbor_tasks_from_dir(Path(launch["config"]["tasks_dir"]))
    hashes = {task.task_name: _task_digest(task) for task in tasks}
    if hashes != launch["source_manifest"]["task_hashes"]:
        raise ValueError("The frozen task manifest changed")
    selected = [task for task in tasks if task.task_name in config.task_names]
    if len(selected) != len(config.task_names):
        raise ValueError("A selected task is absent from the frozen manifest")
    evidence = {}
    for filename in config.validation_files:
        path = Path(filename)
        text = path.read_text()
        record = json.loads(text)
        if record["task"] in evidence:
            raise ValueError("Resource validation is ambiguous")
        evidence[record["task"]] = {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "record": record,
        }
    if set(evidence) != set(config.task_names):
        raise ValueError("Every selected task needs one explicit resource validation record")
    for task in selected:
        validate_resource_record(
            evidence[task.task_name]["record"],
            task_name=task.task_name,
            task_hash=hashes[task.task_name],
        )
    policy = json.dumps(
        {
            "backend": "modal",
            "tasks": sorted(config.task_names),
            "memory_mb": 16384,
            "cpu": 4.0,
            "build_parallelism": 4,
        },
        sort_keys=True,
    )
    configs: dict[str, CLIConfig] = {}
    pending: list[tuple[str, HarborTask, Path]] = []
    completed: dict[str, list[OpenAITaskResult]] = {model: [] for model in MODELS}
    for model in MODELS:
        source = root / model.split("/")[-1]
        saved = json.loads((source / "eval_identity.json").read_text())
        if saved["tasks"] != hashes or saved["identity"]["config"]["model_name"] != model:
            raise ValueError("Original model identity differs from the frozen manifest")
        values = {
            **saved["identity"]["config"],
            "sandbox_resource_policy": policy,
            "sandbox_build_parallelism": 4,
        }
        evaluation = CLIConfig(**values)
        if evaluation.grader_timeout != 900 or evaluation.chat_provider != "nebius":
            raise ValueError("Original evaluation provider/deadline differs")
        configs[model] = evaluation
        destination = phase / model.split("/")[-1]
        prepare_eval_state(destination, values, selected, evaluator="nebius-modal-resource-phase")
        for task in selected:
            original = source / task.task_name
            if (original / "attempt_started.json").exists() or (
                original / "results.jsonl"
            ).exists():
                raise ValueError("Resource phase cannot resample an original attempt")
            trial = destination / task.task_name
            if (trial / "results.jsonl").exists():
                rows = [
                    json.loads(line)
                    for line in (trial / "results.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                if len(rows) != 1 or rows[0]["task_name"] != task.task_name:
                    raise ValueError("Resource phase result history is ambiguous")
                completed[model].append(OpenAITaskResult(**rows[0]))
            elif (trial / "attempt_started.json").exists():
                raise ValueError("Interrupted resource-phase attempt requires manual review")
            else:
                pending.append((model, task, trial))
    load_env_file(Path(launch["config"]["env_file"]))
    first = configs[MODELS[0]]
    async with AsyncOpenAI(
        api_key=os.environ[first.api_key_env], base_url=first.base_url, timeout=None
    ) as client:
        catalog = await client.models.list()
        if not set(MODELS).issubset({model.id for model in catalog.data}):
            raise ValueError("An exact requested model is unavailable")
        launch_record = {
            "task_names": config.task_names,
            "validation": evidence,
            "resource_policy": json.loads(policy),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        if not (phase / "launch.json").exists():
            write_json(phase / "launch.json", launch_record)
        write_json(
            phase / "launches" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json",
            launch_record,
        )

        async def reject_other_backend(env_dir: Path, timeout: int) -> SandboxInterface:
            raise ValueError("Unexpected task attempted outside the fixed Modal resource policy")

        factory = create_resource_sandbox_factory(
            reject_other_backend, config.task_names, memory_mb=16384, cpu=4.0, build_parallelism=4
        )
        slots, lock = asyncio.Semaphore(config.max_concurrency), asyncio.Lock()

        async def evaluate(pair: tuple[str, HarborTask, Path]) -> None:
            model, task, trial = pair
            async with slots:
                trial.mkdir(parents=True, exist_ok=True)
                write_json(
                    trial / "attempt_started.json",
                    {
                        "model": model,
                        "task": task.task_name,
                        "started_at": datetime.now(UTC).isoformat(),
                        "commit": launch_record["commit"],
                    },
                )
                result = await evaluate_task(
                    task,
                    client,
                    factory,
                    configs[model],
                    trial,
                    lock,
                    before_grading=partial(capture_candidate, task=task, results_dir=trial),
                )
                completed[model].append(result)
                write_json(
                    phase / "status.json",
                    {name: summarize(rows, len(selected)) for name, rows in completed.items()},
                )

        pending.sort(key=lambda pair: (pair[1].task_name, MODELS.index(pair[0])))
        await asyncio.gather(*(evaluate(pair) for pair in pending))
        write_json(
            phase / "status.json",
            {name: summarize(rows, len(selected)) for name, rows in completed.items()},
        )


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(ResourcePhaseConfig)))
