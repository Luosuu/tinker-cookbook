"""One bounded recovery after the original sweep releases its controller lock.

Retain original failed attempts. A reviewed read-only prefix is replayed locally,
not sampled again. Each original model-task permits only one recovery marker.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import chz
import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest, prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HARBOR_SYSTEM_PROMPT,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import (
    ImportedCacheFactory,
    validated,
    write_json,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import (
    BASH_TOOL,
    CLIConfig,
    OpenAITaskResult,
    evaluate_task,
    load_env_file,
)
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_recovery import (
    PrefixReplayTransport,
    ReadOnlyPrefix,
    recovery_accounting,
    validate_recovery_source,
)


@chz.chz
class RecoveryConfig:
    source_root: str
    model: str
    task_name: str
    # Empty only when the original provider rejected before any generation.
    approved_readonly_commands: tuple[str, ...] = ()


async def main(config: RecoveryConfig) -> None:
    root = Path(config.source_root)
    with (root / "controller.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Original controller still owns the shared inference slots"
            ) from error
        await recover(config)


async def recover(config: RecoveryConfig) -> None:
    root = Path(config.source_root)
    launch = json.loads((root / "launch.json").read_text())
    source_dir = root / config.model.split("/")[-1]
    identity_text = (source_dir / "eval_identity.json").read_text()
    saved = json.loads(identity_text)
    evaluation = CLIConfig(**saved["identity"]["config"])
    if config.model != evaluation.model_name or evaluation.chat_provider != "nebius":
        raise ValueError("Recovery model/provider differs from original identity")
    tasks = load_harbor_tasks_from_dir(Path(launch["config"]["tasks_dir"]))
    hashes = {task.task_name: _task_digest(task) for task in tasks}
    if hashes != saved["tasks"] or hashes != launch["source_manifest"]["task_hashes"]:
        raise ValueError("Recovery snapshot differs from the original full task manifest")
    task = next(task for task in tasks if task.task_name == config.task_name)
    if task.task_name in launch["config"]["modal_task_names"]:
        raise ValueError("This recovery supports only the unchanged ConTree resource policy")
    if not validated(task, Path(launch["config"]["validation_dir"])):
        raise ValueError("Original verifier gate is not passing")
    original_dir = source_dir / task.task_name
    result_text = (original_dir / "results.jsonl").read_text()
    rows = [json.loads(line) for line in result_text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("Recovery requires exactly one original attempt")
    original = OpenAITaskResult(**rows[0])
    if original.task_name != task.task_name:
        raise ValueError("Original result belongs to another task")
    prefix = None
    if original.turns_used:
        prefix = ReadOnlyPrefix.from_transcript(
            (original_dir / f"{task.task_name}.json").read_text(),
            approved_commands=config.approved_readonly_commands,
        )
    validate_recovery_source(original, prefix)
    destination = root / "recoveries" / config.model.split("/")[-1] / task.task_name
    if destination.exists():
        raise ValueError("A recovery already exists; interrupted attempts require manual review")
    load_env_file(Path(launch["config"]["env_file"]))
    key = os.environ[evaluation.api_key_env]
    # Read-only availability preflight, without changing model or generating text.
    async with AsyncOpenAI(api_key=key, base_url=evaluation.base_url, timeout=None) as check:
        models = await check.models.list()
        if config.model not in {model.id for model in models.data}:
            raise ValueError("The exact original model remains unavailable")
    prepare_eval_state(
        destination, saved["identity"]["config"], [task], evaluator="nebius-prefix-recovery"
    )
    provenance = {
        "source_result": str(original_dir / "results.jsonl"),
        "source_result_sha256": hashlib.sha256(result_text.encode()).hexdigest(),
        "source_identity_sha256": hashlib.sha256(identity_text.encode()).hexdigest(),
        "task_hash": hashes[task.task_name],
        "original_error": original.error,
        "approved_readonly_commands": config.approved_readonly_commands,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
    }
    write_json(destination / "attempt_started.json", provenance)
    transport = None
    if prefix is not None:
        first: dict[str, object] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": HARBOR_SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {k: v for k, v in BASH_TOOL.items() if k != "type"},
                }
            ],
            "max_tokens": min(evaluation.max_tokens, evaluation.max_sampled_tokens),
            "reasoning_effort": evaluation.reasoning_effort,
        }
        if evaluation.temperature is not None:
            first["temperature"] = evaluation.temperature
        transport = PrefixReplayTransport(
            prefix=prefix, expected_first_request=first, transport=httpx.AsyncHTTPTransport()
        )
    factory = ImportedCacheFactory(
        destination / "contree_images.json",
        timeout=evaluation.sandbox_timeout,
        runtime_build_parallelism=evaluation.sandbox_build_parallelism,
        allow_network=evaluation.allow_network,
    )
    factory.import_cache(root / "contree_images.json")
    # OpenAI's compatibility adapter supports an injected legacy httpx client.
    async with AsyncOpenAI(
        api_key=key,
        base_url=evaluation.base_url,
        timeout=None,
        max_retries=0,
        http_client=cast(
            DefaultAsyncHttpxClient, httpx.AsyncClient(transport=transport, timeout=None)
        ),
    ) as client:
        result = await evaluate_task(task, client, factory, evaluation, destination, asyncio.Lock())
    write_json(
        destination / "recovery.json",
        {
            **provenance,
            "replay": transport.provenance if transport else None,
            "usage": recovery_accounting(
                original, result, prefix_replayed=transport.replayed if transport else False
            ),
            "complete": result.error is None,
            "error": result.error,
        },
    )


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(RecoveryConfig)))
