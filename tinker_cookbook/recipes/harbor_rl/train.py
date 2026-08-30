"""CLI entry point for Harbor RL training."""

import logging
import os
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborDatasetBuilder,
    HarborTask,
    SandboxFactory,
    harbor_termination_policy,
)
from tinker_cookbook.recipes.harbor_rl.heldout_eval import BoundedHeldOutEvaluatorBuilder
from tinker_cookbook.rl.rollout_presets import default_rollout_strategy_for_model
from tinker_cookbook.rl.rollout_strategy import RolloutStrategy, rollout_strategy_from_config
from tinker_cookbook.rl.train import AsyncConfig, Config, main

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for Harbor RL training."""

    # Model configuration
    model_name: str = "moonshotai/Kimi-K2.6"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    max_tokens: int = 8192
    temperature: float = 1.0

    # Environment configuration
    max_turns: int = 10
    sandbox_timeout: int = 3600
    command_timeout: int = 120
    grader_timeout: int = 60
    max_trajectory_tokens: int = 32 * 1024
    max_tool_calls: int | None = None
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    thinking_effort: float | None = None

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 8
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    # Drop groups whose trajectories all earned the same reward (no gradient signal).
    remove_constant_reward_groups: bool = False
    # Raise on grading infrastructure failures instead of scoring them 0.0, so the
    # rollout strategy can retry or drop the group rather than treating a crashed
    # verifier as a model failure.
    raise_on_grading_error: bool = False
    # Tolerance for rollout errors (sandbox crashes, grading failures). None uses the
    # model default (MinViableGroup for Inkling, FailFast otherwise); False is
    # FailFast; True is RetryOnFailure. Applies to training and held-out eval alike.
    rollout_error_tolerance: bool | RolloutStrategy | None = None

    # Held-out evaluation. eval_size=0 keeps the legacy in-sample eval; with
    # eval_size > 0 the tasks are split deterministically by seed and the held-out
    # slice is rolled out under a concurrency limit, so peak sandbox concurrency
    # stays at heldout_max_concurrent_groups * eval_group_size.
    eval_size: int = 0
    eval_group_size: int = 1
    heldout_max_concurrent_groups: int = 2
    split_seed: int = 0

    # Logging / eval / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Weave tracing of per-episode grading; requires WANDB_API_KEY.
    trace_weave: bool = False
    weave_project: str | None = None
    eval_every: int = 5
    save_every: int = 5

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Async rollout configuration
    max_steps_off_policy: int | None = None

    max_steps: int | None = None


async def cli_main(
    cli_config: CLIConfig,
    tasks: list[HarborTask],
    sandbox_factory: SandboxFactory | None = None,
) -> None:
    if cli_config.trace_weave:
        weave_project = cli_config.weave_project or cli_config.wandb_project
        if weave_project is None:
            raise ValueError(
                "trace_weave=True requires weave_project (or wandb_project) to be set."
            )
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("trace_weave=True requires WANDB_API_KEY to be set.")
        import weave

        weave.init(weave_project)
        logger.info("Weave tracing enabled for project %s", weave_project)

    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"harbor-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/harbor_rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name
    max_generation_tokens = (
        cli_config.max_generation_tokens
        if cli_config.max_generation_tokens is not None
        else cli_config.max_tokens
    )

    dataset_builder = HarborDatasetBuilder(
        tasks=tasks,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        max_turns=cli_config.max_turns,
        sandbox_timeout=cli_config.sandbox_timeout,
        command_timeout=cli_config.command_timeout,
        grader_timeout=cli_config.grader_timeout,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_tool_calls=cli_config.max_tool_calls,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=cli_config.context_overflow_reward,
        sandbox_factory=sandbox_factory,
        thinking_effort=cli_config.thinking_effort,
        raise_on_grading_error=cli_config.raise_on_grading_error,
        eval_size=cli_config.eval_size,
        eval_group_size=cli_config.eval_group_size,
        seed=cli_config.split_seed,
    )

    # When a held-out split is configured, the dataset builder returns None for the
    # test slot (so rl.train does not auto-wrap an in-sample eval) and the held-out
    # tasks are measured by an evaluator that bounds sandbox concurrency instead.
    # The evaluator gets the same rollout strategy the trainer uses: without it,
    # a sandbox or grading failure during eval propagates and kills the whole run
    # (rather than being retried or dropped as a single group).
    evaluator_builders = []
    held_out_builders = dataset_builder.make_held_out_env_group_builders()
    if held_out_builders:
        evaluator_builders.append(
            BoundedHeldOutEvaluatorBuilder(
                env_group_builders=held_out_builders,
                max_tokens=cli_config.max_tokens,
                max_concurrent_groups=cli_config.heldout_max_concurrent_groups,
                strategy=default_rollout_strategy_for_model(cli_config.model_name)
                if cli_config.rollout_error_tolerance is None
                else rollout_strategy_from_config(cli_config.rollout_error_tolerance),
            )
        )

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_harbor_rl",
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        rollout_error_tolerance=cli_config.rollout_error_tolerance,
        # Trainer-side half of the rollout config. do_group_rollout applies the
        # clamp from Config.effective_termination(), which otherwise falls back
        # to the model preset and re-zeros passing-but-long trajectories even
        # though the env-side config drops MAX_TURNS.
        termination=harbor_termination_policy(),
        evaluator_builders=evaluator_builders,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    await main(config)
