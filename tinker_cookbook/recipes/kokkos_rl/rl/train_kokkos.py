"""Launch Harbor RL training on the exported Kokkos coding tasks.

This is the training counterpart to ``eval_kokkos.py``. It loads the tasks that
``export_harbor.py`` wrote to a directory and hands them to the shared Harbor RL
training loop, so no Kokkos-specific training code is duplicated.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    SandboxFactory,
    default_sandbox_factory,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig as HarborCLIConfig
from tinker_cookbook.recipes.harbor_rl.train import cli_main
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import (
    DEFAULT_CONTREE_CACHE_PATH,
    load_env_file,
    select_tasks,
)
from tinker_cookbook.rl.rollout_strategy import RolloutStrategy


@chz.chz
class CLIConfig:
    """Kokkos RL training configuration.

    Defaults use the full clean-room benchmark with a fixed held-out split, the
    long-context GPT-OSS variant, and a generous budget for multi-file C++ fixes.
    """

    model_name: str = "openai/gpt-oss-120b:peft:131072"
    tasks_dir: str = "data/kokkos/SWE-kokkos-bench-v2"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    max_tokens: int = 16384
    temperature: float = 1.0

    # Environment configuration (matches the long-context eval budget).
    max_turns: int = 20
    sandbox_timeout: int = 3600
    # Kokkos' serial unit-test targets take about 400 seconds on ConTree even
    # without competing rollouts. Leave headroom for backend load variance.
    command_timeout: int = 900
    grader_timeout: int = 180
    max_trajectory_tokens: int = 112 * 1024
    max_tool_calls: int | None = 80
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    thinking_effort: float | None = None

    # Sandbox backend (mirrors ``eval_kokkos.py``). ConTree is the economical
    # default; Modal remains an explicit fallback for backend failures.
    sandbox_backend: str = "contree"
    contree_cache_path: str | None = None
    sandbox_build_parallelism: int | None = None
    # Image preparation may use the network, but rollout sandboxes default to no
    # network so agents cannot retrieve the merged PR or other solution material.
    allow_network: bool = False
    env_file: str = ".env"

    # Training hyperparameters.
    group_size: int = 4
    groups_per_batch: int = 4
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    remove_constant_reward_groups: bool = True
    raise_on_grading_error: bool = True
    rollout_error_tolerance: bool | RolloutStrategy | None = None

    # Fixed held-out evaluation defaults used by the clean-room Kokkos runs.
    eval_size: int = 20
    eval_group_size: int = 4
    heldout_max_concurrent_groups: int = 2
    split_seed: int = 0

    # Restrict the task pool before the held-out split (comma-separated names).
    task_names: str | None = None

    # Logging / eval / checkpoints.
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    trace_weave: bool = False
    weave_project: str | None = None
    eval_every: int = 5
    save_every: int = 5
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Service / async configuration.
    base_url: str | None = None
    max_steps_off_policy: int | None = None
    max_steps: int | None = None


def _to_harbor_config(cli_config: CLIConfig) -> HarborCLIConfig:
    """Translate the Kokkos config into the shared Harbor training config."""
    return HarborCLIConfig(
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        max_turns=cli_config.max_turns,
        sandbox_timeout=cli_config.sandbox_timeout,
        command_timeout=cli_config.command_timeout,
        grader_timeout=cli_config.grader_timeout,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_tool_calls=cli_config.max_tool_calls,
        max_generation_tokens=cli_config.max_generation_tokens,
        context_overflow_reward=cli_config.context_overflow_reward,
        thinking_effort=cli_config.thinking_effort,
        group_size=cli_config.group_size,
        groups_per_batch=cli_config.groups_per_batch,
        learning_rate=cli_config.learning_rate,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        raise_on_grading_error=cli_config.raise_on_grading_error,
        rollout_error_tolerance=cli_config.rollout_error_tolerance,
        eval_size=cli_config.eval_size,
        eval_group_size=cli_config.eval_group_size,
        heldout_max_concurrent_groups=cli_config.heldout_max_concurrent_groups,
        split_seed=cli_config.split_seed,
        log_path=cli_config.log_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        trace_weave=cli_config.trace_weave,
        weave_project=cli_config.weave_project,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        behavior_if_log_dir_exists=cli_config.behavior_if_log_dir_exists,
        base_url=cli_config.base_url,
        max_steps_off_policy=cli_config.max_steps_off_policy,
        max_steps=cli_config.max_steps,
    )


def _build_sandbox_factory(cli_config: CLIConfig) -> SandboxFactory:
    """Resolve the sandbox backend, mirroring ``eval_kokkos.main``.

    ConTree prepares each task's exported Dockerfile once and branches isolated
    sessions from the resulting immutable image. Prepared image UUIDs are keyed
    by Dockerfile digest, so the cache defaults to a stable shared path rather
    than a per-run directory: point ``contree_cache_path`` at the cache written
    by an earlier evaluation to reuse images instead of rebuilding them.
    """
    if cli_config.sandbox_backend == "modal":
        return functools.partial(
            default_sandbox_factory,
            allow_network=cli_config.allow_network,
        )
    if cli_config.sandbox_backend == "contree":
        from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory

        cache_path = Path(cli_config.contree_cache_path or DEFAULT_CONTREE_CACHE_PATH)
        return ContreeDockerfileSandboxFactory(
            cache_path=cache_path,
            timeout=cli_config.sandbox_timeout,
            runtime_build_parallelism=cli_config.sandbox_build_parallelism,
            allow_network=cli_config.allow_network,
        )
    raise ValueError(f"unknown sandbox_backend: {cli_config.sandbox_backend!r}")


async def main(cli_config: CLIConfig) -> None:
    load_env_file(Path(cli_config.env_file))
    tasks_dir = Path(cli_config.tasks_dir)
    tasks = select_tasks(load_harbor_tasks_from_dir(tasks_dir), cli_config.task_names)
    if not tasks:
        raise ValueError(f"No Harbor tasks found under {tasks_dir}")
    sandbox_factory = _build_sandbox_factory(cli_config)
    print(
        f"Training {cli_config.model_name} on {len(tasks)} Kokkos tasks from "
        f"{tasks_dir} (group_size={cli_config.group_size}, "
        f"groups_per_batch={cli_config.groups_per_batch}, "
        f"sandbox_backend={cli_config.sandbox_backend}, "
        f"eval_size={cli_config.eval_size})"
    )
    await cli_main(
        _to_harbor_config(cli_config),
        tasks,
        sandbox_factory=sandbox_factory,
    )


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(CLIConfig)))
