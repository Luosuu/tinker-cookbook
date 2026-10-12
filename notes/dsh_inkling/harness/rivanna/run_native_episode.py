"""Run one native Chat Completions episode on a Rivanna Apptainer sandbox.

The configuration is the one in `harness/native/run_native_contree.py`; only the sandbox
differs. Both arms of a Rivanna block open the sandbox through `apptainer.open_sandbox`, so
images, preflight, thread settings and grading are shared and the harness is the only
difference. Writes `summary.json` in the same layout as a DSH episode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def native_config(budget: dict):
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import CLIConfig

    return CLIConfig(
        model_name="thinkingmachines/Inkling-Small:peft:262144",
        api_mode="chat",
        chat_provider="tinker",
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        api_key_env="TINKER_API_KEY",
        thinking_effort=0.9,
        reasoning_effort="high",
        temperature=1.0,
        max_turns=budget["max_model_requests"],
        max_tokens=16384,
        max_sampled_tokens=budget["max_sampled_tokens"],
        max_input_tokens=budget["max_input_tokens"],
        max_context_tokens=budget["max_context_tokens"],
        max_tool_calls=budget["max_tool_calls"],
        max_tool_output_chars=16384,
        max_cost_usd_per_task=None,
        estimate_cost=False,
        sandbox_timeout=budget["sandbox_timeout"],
        command_timeout=budget["command_timeout"],
        grader_timeout=budget["grader_timeout"],
        max_concurrency=1,
        allow_network=False,
    )


async def run(
    *,
    task: str,
    output: Path,
    tasks_root: Path,
    code_root: Path,
    env_file: Path,
    budget: dict,
) -> dict:
    sys.path.insert(0, str(code_root.resolve()))
    from notes.dsh_inkling.harness.rivanna.apptainer import load_guarded_task, open_sandbox
    from tinker_cookbook.recipes.kokkos_rl.chat_inference import create_chat_client
    from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import evaluate_task
    from tinker_cookbook.utils.ml_log import dump_config

    load_env_file(env_file)
    harbor_task = load_guarded_task(tasks_root, task)
    config = native_config(budget)
    max_infra_retries = budget["native_max_infra_retries"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "harness": "native",
                "sandbox": "apptainer",
                "sif_path": (harbor_task.task_dir / "environment/sif.path").read_text().strip(),
                "task": task,
                "max_infra_retries": max_infra_retries,
                "eval_config": dump_config(config),
            },
            indent=2,
        )
    )

    async def factory(_env_dir: Path, timeout: int):
        return await open_sandbox(harbor_task, timeout, hidden_paths=(str(env_file.resolve()),))

    async def capture(sandbox) -> None:
        await capture_candidate(sandbox, task=harbor_task, results_dir=output)

    started = time.monotonic()
    client = create_chat_client(config.base_url, config.api_key_env)
    attempts: list[dict] = []
    try:
        # The direct endpoint intermittently answers 502; retrying only infrastructure
        # errors raises the native score, which makes "DSH beats native" a stricter bar.
        for _ in range(max_infra_retries + 1):
            result = asdict(
                await evaluate_task(
                    harbor_task,
                    client,
                    factory,
                    config,
                    output,
                    asyncio.Lock(),
                    before_grading=capture,
                )
            )
            attempts.append(result)
            if result["error"] is None:
                break
    finally:
        await client.close()
    with (output / "attempts.jsonl").open("w") as stream:
        for attempt in attempts:
            stream.write(json.dumps(attempt) + "\n")
    summary = {
        **attempts[-1],
        "harness": "native",
        "harness_status": "finished" if attempts[-1]["error"] is None else "error",
        "attempts": len(attempts),
        "time_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--budget", action="append", default=[], help="key=value override of EPISODE_BUDGET"
    )
    args = parser.parse_args()
    from notes.dsh_inkling.harness.rivanna.budget import resolve_budget

    args.budget = resolve_budget(args.budget)
    print(json.dumps(asyncio.run(run(**vars(args))), indent=2, default=str))


if __name__ == "__main__":
    main()
