"""Evaluate a Tinker sampling model on the exported Kokkos Harbor tasks."""

from __future__ import annotations

import asyncio
import functools
import os
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
    tasks_dir: str = "data/kokkos/SWE-kokkos-bench-v2"
    output_path: str = "notes/experiments/kokkos_rl/evals"
    checkpoint_url: str | None = None
    env_file: str = ".env"

    max_turns: int = 20
    max_tokens: int = 16384
    temperature: float = 1.0
    sandbox_timeout: int = 3600
    # Kokkos' serial unit-test targets take about 400 seconds on ConTree even
    # without competing rollouts. Leave headroom for backend load variance.
    command_timeout: int = 900
    grader_timeout: int = 180
    sandbox_build_parallelism: int | None = None
    max_tasks: int | None = None
    max_concurrency: int = 2
    task_names: str | None = None
    max_trajectory_tokens: int = 112 * 1024
    max_sampled_tokens: int = 64 * 1024
    max_tool_calls: int = 40

    base_url: str | None = None
    renderer_name: str | None = None
    thinking_effort: float | None = None
    sandbox_backend: str = "modal"
    contree_cache_path: str | None = None
    # Rollout sandboxes get no network interface when False. Tasks derive from
    # merged PRs, so an evaluation with network open measures answer-lookup as
    # much as problem-solving; keep this matched to how the model was trained.
    allow_network: bool = False
    num_samples: int = 1
    pass_at_k: str = "1"
    resume_dir: str | None = None
    max_infra_retries: int = 2


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def print_summary(results: list[TaskResult]) -> None:
    passed = sum(result.error is None and result.reward > 0 for result in results)
    errored = sum(result.error is not None for result in results)
    total = len(results)
    print(
        f"kokkos: total={total} pass={passed} fail={total - passed - errored} "
        f"error={errored} pass_rate={passed / total if total else 0.0:.1%}"
    )


def select_tasks(tasks: list, task_names: str | None) -> list:
    """Select a deterministic comma-separated task subset for incremental evals."""

    if task_names is None:
        return tasks
    wanted = {name.strip() for name in task_names.split(",") if name.strip()}
    selected = [task for task in tasks if task.task_name in wanted]
    missing = wanted - {task.task_name for task in selected}
    if missing:
        raise ValueError(f"unknown task names: {sorted(missing)}")
    return selected


async def main(cli_config: CLIConfig) -> None:
    load_env_file(Path(cli_config.env_file))
    tasks = select_tasks(
        load_harbor_tasks_from_dir(Path(cli_config.tasks_dir)), cli_config.task_names
    )
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
        sandbox_build_parallelism=cli_config.sandbox_build_parallelism,
        max_tasks=cli_config.max_tasks,
        max_concurrency=cli_config.max_concurrency,
        base_url=cli_config.base_url,
        renderer_name=cli_config.renderer_name,
        thinking_effort=cli_config.thinking_effort,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_sampled_tokens=cli_config.max_sampled_tokens,
        max_tool_calls=cli_config.max_tool_calls,
        num_samples=cli_config.num_samples,
        pass_at_k=cli_config.pass_at_k,
        resume_dir=cli_config.resume_dir,
        max_infra_retries=cli_config.max_infra_retries,
    )
    print(
        f"Running {len(tasks)} Kokkos tasks with model={eval_config.model_name}, "
        f"temperature={eval_config.temperature}, max_tokens={eval_config.max_tokens}, "
        f"thinking_effort={eval_config.thinking_effort}"
    )
    if cli_config.sandbox_backend == "modal":
        sandbox_factory = functools.partial(
            default_sandbox_factory,
            allow_network=cli_config.allow_network,
        )
    elif cli_config.sandbox_backend == "contree":
        from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory

        cache_path = Path(
            cli_config.contree_cache_path or Path(cli_config.output_path) / "contree_images.json"
        )
        sandbox_factory = ContreeDockerfileSandboxFactory(
            cache_path=cache_path,
            timeout=cli_config.sandbox_timeout,
            runtime_build_parallelism=cli_config.sandbox_build_parallelism,
            allow_network=cli_config.allow_network,
        )
    else:
        raise ValueError(f"unknown sandbox_backend: {cli_config.sandbox_backend!r}")
    results = await run_eval(eval_config, tasks, sandbox_factory=sandbox_factory)
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(CLIConfig)))
