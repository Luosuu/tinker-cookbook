"""RL training entrypoint: Inkling-Small on Nemotron web-search MCQA.

Trains Inkling-Small with Serper search+browse tools and a \\boxed{LETTER}
reward, using the cookbook's synchronous RL loop.

Example (smoke test, 2 tiny steps):
    uv run python -m tinker_cookbook.recipes.search_tool.nemotron_train \
        batch_size=4 group_size=4 max_steps=2 n_examples=64 eval_every=0

Example (real run):
    uv run python -m tinker_cookbook.recipes.search_tool.nemotron_train \
        batch_size=128 group_size=8 learning_rate=2e-5 wandb_project=nemotron_mcqa

Requires SERPER_API_KEY and TINKER_API_KEY (loaded from repo-root .env if present).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.search_tool.nemotron_env import NemotronDatasetBuilder
from tinker_cookbook.rl import train


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


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
    eval_every: int = 0
    save_every: int = 20
    max_steps: int | None = None

    # Rollout / tools
    max_turns: int = 16
    max_tool_calls: int = 30
    max_trajectory_tokens: int = 96 * 1024
    n_results: int = 5
    browse_chars: int = 8000
    format_coef: float = 0.1
    n_examples: int | None = None  # cap dataset size (smoke tests)

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    weave_project: str | None = None  # None -> use wandb_project; set to enable Weave rollout traces
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    base_url: str | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    _load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    if not os.environ.get("SERPER_API_KEY"):
        raise SystemExit("SERPER_API_KEY not set (checked env and repo-root .env).")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (checked env and repo-root .env).")

    # Weave: trace rollout trajectories into the same wandb project. Requires
    # WANDB_API_KEY. weave.init() must run before any @weave.op() fires, i.e.
    # before training starts. If it is not called, the reward op is inert.
    weave_project = cli_config.weave_project or cli_config.wandb_project
    trace_weave = weave_project is not None
    if trace_weave:
        if not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("weave tracing requested but WANDB_API_KEY is not set.")
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

    config = train.Config(
        model_name=cli_config.model_name,
        recipe_name="recipe_nemotron_mcqa",
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
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
