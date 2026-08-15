"""Evaluate a Tinker sampling model on the exported Kokkos Harbor tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, TaskResult, run_eval
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    default_sandbox_factory,
    load_harbor_tasks_from_dir,
)


@chz.chz
class CLIConfig:
    model_name: str = "openai/gpt-oss-120b:peft:131072"
    tasks_dir: str = "data/kokkos/phase0/harbor"
    output_path: str = "notes/experiments/kokkos_phase0/baselines"
    checkpoint_url: str | None = None

    max_turns: int = 20
    max_tokens: int = 16384
    temperature: float = 1.0
    sandbox_timeout: int = 3600
    command_timeout: int = 180
    grader_timeout: int = 180
    max_tasks: int | None = None
    max_trajectory_tokens: int = 112 * 1024
    max_sampled_tokens: int = 64 * 1024
    max_tool_calls: int = 40

    base_url: str | None = None
    renderer_name: str | None = None
    thinking_effort: float | None = None


def print_summary(results: list[TaskResult]) -> None:
    passed = sum(result.error is None and result.reward > 0 for result in results)
    errored = sum(result.error is not None for result in results)
    total = len(results)
    print(
        f"kokkos: total={total} pass={passed} fail={total - passed - errored} "
        f"error={errored} pass_rate={passed / total if total else 0.0:.1%}"
    )


async def main(cli_config: CLIConfig) -> None:
    tasks = load_harbor_tasks_from_dir(Path(cli_config.tasks_dir))
    eval_config = EvalConfig(
        model_name=cli_config.model_name,
        checkpoint_url=cli_config.checkpoint_url,
        output_path=str(Path(cli_config.output_path) / cli_config.model_name.replace("/", "-")),
        max_turns=cli_config.max_turns,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        sandbox_timeout=cli_config.sandbox_timeout,
        command_timeout=cli_config.command_timeout,
        grader_timeout=cli_config.grader_timeout,
        max_tasks=cli_config.max_tasks,
        base_url=cli_config.base_url,
        renderer_name=cli_config.renderer_name,
        thinking_effort=cli_config.thinking_effort,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_sampled_tokens=cli_config.max_sampled_tokens,
        max_tool_calls=cli_config.max_tool_calls,
    )
    print(
        f"Running {len(tasks)} Kokkos tasks with model={eval_config.model_name}, "
        f"temperature={eval_config.temperature}, max_tokens={eval_config.max_tokens}, "
        f"thinking_effort={eval_config.thinking_effort}"
    )
    results = await run_eval(eval_config, tasks, sandbox_factory=default_sandbox_factory)
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(CLIConfig)))
