"""Launch Harbor RL training on the exported Kokkos coding tasks.

This is the training counterpart to ``eval_kokkos.py``. It loads the tasks that
``export_harbor.py`` wrote to a directory and hands them to the shared Harbor RL
training loop, so no Kokkos-specific training code is duplicated.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    default_sandbox_factory,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig as HarborCLIConfig
from tinker_cookbook.recipes.harbor_rl.train import cli_main


@chz.chz
class CLIConfig:
    """Kokkos RL training configuration.

    Defaults track the Phase 0 baseline in ``eval_kokkos.py``: the long-context
    GPT-OSS variant, a generous turn/token budget for multi-file C++ fixes, and
    a LoRA learning rate suited to the coding-RL setup.
    """

    model_name: str = "openai/gpt-oss-120b:peft:131072"
    tasks_dir: str = "data/kokkos/phase0/harbor"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    max_tokens: int = 16384
    temperature: float = 1.0

    # Environment configuration (matches the Phase 0 eval budget).
    max_turns: int = 20
    sandbox_timeout: int = 3600
    command_timeout: int = 180
    grader_timeout: int = 180
    max_trajectory_tokens: int = 112 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    thinking_effort: float | None = None

    # Training hyperparameters.
    group_size: int = 4
    groups_per_batch: int = 8
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging / eval / checkpoints.
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 5
    save_every: int = 5

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
        max_generation_tokens=cli_config.max_generation_tokens,
        context_overflow_reward=cli_config.context_overflow_reward,
        thinking_effort=cli_config.thinking_effort,
        group_size=cli_config.group_size,
        groups_per_batch=cli_config.groups_per_batch,
        learning_rate=cli_config.learning_rate,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        log_path=cli_config.log_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        base_url=cli_config.base_url,
        max_steps_off_policy=cli_config.max_steps_off_policy,
        max_steps=cli_config.max_steps,
    )


async def main(cli_config: CLIConfig) -> None:
    tasks_dir = Path(cli_config.tasks_dir)
    tasks = load_harbor_tasks_from_dir(tasks_dir)
    if not tasks:
        raise ValueError(f"No Harbor tasks found under {tasks_dir}")
    print(
        f"Training {cli_config.model_name} on {len(tasks)} Kokkos tasks from "
        f"{tasks_dir} (group_size={cli_config.group_size}, "
        f"groups_per_batch={cli_config.groups_per_batch})"
    )
    await cli_main(
        _to_harbor_config(cli_config),
        tasks,
        sandbox_factory=default_sandbox_factory,
    )


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(CLIConfig)))
