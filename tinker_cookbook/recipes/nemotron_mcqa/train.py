"""RL training entrypoint: Inkling-Small on Nemotron web-search MCQA.

Trains Inkling-Small with Serper search+browse tools and a \\boxed{LETTER}
reward, using the cookbook's synchronous RL loop.

Example (smoke test, 2 tiny steps):
    uv run python -m tinker_cookbook.recipes.nemotron_mcqa.train \
        batch_size=4 group_size=4 max_steps=2 n_examples=64 eval_every=0

Example (real run):
    uv run python -m tinker_cookbook.recipes.nemotron_mcqa.train \
        batch_size=128 group_size=8 learning_rate=2e-5 wandb_project=nemotron_mcqa

Requires SERPER_API_KEY and TINKER_API_KEY (loaded from repo-root .env if present).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.nemotron_mcqa.common import (
    DEFAULT_SPLIT_DATASET,
    load_dotenv,
    repo_root_dotenv,
    require_provider_key,
)
from tinker_cookbook.recipes.nemotron_mcqa.env import (
    NemotronDatasetBuilder,
    NemotronValEvaluatorBuilder,
)
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model
    model_name: str = "thinkingmachines/Inkling-Small"
    lora_rank: int = 32
    renderer_name: str | None = None  # None -> auto (tml_v0 for Inkling)

    # Training
    learning_rate: float = 2e-5  # Inkling has no default LR; sweep this.
    batch_size: int = 128
    group_size: int = 8
    max_tokens: int = 8192
    seed: int = 0
    # eval_every=None -> defaults to save_every so validation runs at every
    # checkpoint. Set 0 to disable, or a positive int to override.
    eval_every: int | None = None
    save_every: int = 20
    max_steps: int | None = None

    # Data (persisted train/validation split repo)
    dataset_name: str = DEFAULT_SPLIT_DATASET

    # Web search/browse provider: serper | tavily | parallel
    provider: str = "serper"

    # Rollout / tools
    max_turns: int = 16
    max_tool_calls: int = 30
    max_trajectory_tokens: int = 96 * 1024
    n_results: int = 5
    browse_chars: int = 8000
    format_coef: float = 0.1
    n_examples: int | None = None  # cap train split size (smoke tests)

    # Validation (held-out eval run every eval_every steps, i.e. at each checkpoint)
    n_val: int | None = None  # None -> whole validation split; 0 disables eval

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Weave rollout-trajectory tracing (opt-in, independent of wandb metrics).
    # Enable with trace_weave=True; traces land in weave_project, defaulting to
    # wandb_project when that is set. Requires WANDB_API_KEY.
    trace_weave: bool = False
    weave_project: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    base_url: str | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    load_dotenv(repo_root_dotenv())
    # Validate the selected provider's key (serper/tavily/parallel) is present.
    require_provider_key(cli_config.provider)
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (checked env and repo-root .env).")

    # Weave: opt-in rollout-trajectory tracing, independent of wandb metrics.
    # weave.init() must run before any @weave.op() fires (i.e. before training);
    # if it is not called, the reward's trace op is inert.
    trace_weave = cli_config.trace_weave
    if trace_weave:
        weave_project = cli_config.weave_project or cli_config.wandb_project
        if weave_project is None:
            raise SystemExit(
                "trace_weave=True requires weave_project (or wandb_project) to be set."
            )
        if not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("trace_weave=True requires WANDB_API_KEY to be set.")
        import weave

        weave.init(weave_project)

    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    builder = NemotronDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        dataset_name=cli_config.dataset_name,
        split="train",
        provider=cli_config.provider,
        max_turns=cli_config.max_turns,
        max_tool_calls=cli_config.max_tool_calls,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        n_results=cli_config.n_results,
        browse_chars=cli_config.browse_chars,
        format_coef=cli_config.format_coef,
        seed=cli_config.seed,
        n_examples=cli_config.n_examples,
        trace_weave=trace_weave,
    )

    model_short = cli_config.model_name.lower().replace("/", "-")
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"nemotron_mcqa_{model_short}_bs{cli_config.batch_size}_"
        f"gs{cli_config.group_size}_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{stamp}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/nemotron_mcqa/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # eval_every defaults to save_every so validation runs at every checkpoint.
    eval_every = (
        cli_config.eval_every if cli_config.eval_every is not None else cli_config.save_every
    )

    # Validation reads the persisted `validation` split (disjoint by construction),
    # so no seed-reconstruction of held-out questions is needed. n_val=None uses
    # the whole validation split.
    evaluator_builders = []
    if eval_every > 0 and cli_config.n_val != 0:
        evaluator_builders.append(
            NemotronValEvaluatorBuilder(
                model_name=cli_config.model_name,
                renderer_name=renderer_name,
                dataset_name=cli_config.dataset_name,
                n_val=cli_config.n_val,
                provider=cli_config.provider,
                max_turns=cli_config.max_turns,
                max_tool_calls=cli_config.max_tool_calls,
                max_trajectory_tokens=cli_config.max_trajectory_tokens,
                max_tokens=cli_config.max_tokens,
                n_results=cli_config.n_results,
                browse_chars=cli_config.browse_chars,
                format_coef=cli_config.format_coef,
                trace_weave=trace_weave,
            )
        )

    config = train.Config(
        model_name=cli_config.model_name,
        recipe_name="recipe_nemotron_mcqa",
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=eval_every,
        evaluator_builders=evaluator_builders,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        max_steps=cli_config.max_steps,
        base_url=cli_config.base_url,
        # Inkling model default already selects agentic() + MinViableGroup, but
        # the env builds its own agentic() rollout_config with a roomier
        # max_turns; termination clamp below matches agentic() semantics.
    )

    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
