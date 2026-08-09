"""Controlled validation eval: score trained checkpoints on a fixed held-out set.

Draws a validation set DISJOINT from the questions training consumed, then runs
the same search+browse agent rollout + \\boxed{} grader against each checkpoint,
on the IDENTICAL questions. This is the apples-to-apples measurement of whether
training improved the model (unlike the training-batch reward, which scores
different questions each step).

Usage:
    uv run python -m tinker_cookbook.recipes.search_tool.nemotron_eval \
        checkpoints='["tinker://.../sampler_weights/000001", "..."]' \
        n_val=24

If checkpoints is empty, pass them on the CLI. base_model is used to render.
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from typing import Any

import chz
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import Message, get_renderer, get_text_content
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.recipes.search_tool.nemotron_env import (
    SYSTEM_PROMPT,
    BoxedLetterReward,
    NemotronDatum,
    SerperTools,
    extract_boxed,
    load_nemotron,
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def build_validation_set(
    n_val: int,
    train_seed: int,
    train_limit: int,
    train_consumed: int,
    val_seed: int,
) -> list[NemotronDatum]:
    """Draw n_val questions guaranteed disjoint from the training questions.

    Reproduces the training shuffle to find which question texts were used,
    then samples validation questions from the rest of the full dataset.
    """
    # Reproduce exactly which questions training consumed.
    train_pool = load_nemotron(limit=train_limit)
    random.Random(train_seed).shuffle(train_pool)
    trained_qs = {d.question for d in train_pool[:train_consumed]}

    # Sample validation from the full dataset, excluding trained questions.
    full = load_nemotron(limit=None)
    candidates = [d for d in full if d.question not in trained_qs]
    random.Random(val_seed).shuffle(candidates)
    return candidates[:n_val]


@chz.chz
class Config:
    checkpoints: list[str] = chz.field(default_factory=list)
    base_model: str = "thinkingmachines/Inkling-Small"
    include_base: bool = True  # also eval the untrained base model as reference
    n_val: int = 24
    val_seed: int = 12345
    # must match the training run so we can exclude its questions
    train_seed: int = 0
    train_limit: int = 64
    train_consumed: int = 12  # batch_size * max_steps
    # rollout knobs (keep close to training)
    max_turns: int = 12
    max_tool_calls: int = 10
    max_trajectory_tokens: int = 96 * 1024
    max_tokens: int = 8192
    n_results: int = 5
    browse_chars: int = 8000
    base_url: str | None = None


async def eval_one(
    q: NemotronDatum, renderer, policy: TinkerTokenCompleter, serper_key: str, cfg: Config
) -> bool | None:
    """Run one rollout, return correct(bool) or None if no answer extracted."""
    tools_obj = SerperTools(serper_key, n_results=cfg.n_results, browse_chars=cfg.browse_chars)
    tools = [tools_obj.search, tools_obj.browse]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[t.to_spec() for t in tools], system_prompt=SYSTEM_PROMPT
    )
    initial_messages = prefix + [Message(role="user", content=q.question)]

    cap: dict[str, Any] = {"final": None}

    async def reward_fn(history: list[Message]) -> tuple[float, dict[str, float]]:
        for m in reversed(history):
            if m.get("role") == "assistant":
                cap["final"] = get_text_content(m)
                break
        return 0.0, {}

    from tinker_cookbook.rl.rollout_presets import agentic

    base = agentic()
    rollout_config = chz.replace(
        base,
        limits=chz.replace(
            base.limits,
            max_turns=cfg.max_turns,
            max_tool_calls=cfg.max_tool_calls,
            max_trajectory_tokens=cfg.max_trajectory_tokens,
        ),
    )
    env = build_agent_tool_env(
        renderer=renderer,
        tools=tools,
        initial_messages=initial_messages,
        reward_fn=reward_fn,
        rollout_config=rollout_config,
    )
    await do_single_rollout(policy, env)
    pred = extract_boxed(cap["final"] or "")
    if pred is None:
        return None
    return pred == q.gold_letter


async def eval_checkpoint(
    label: str,
    model_path: str | None,
    val: list[NemotronDatum],
    service_client: tinker.ServiceClient,
    cfg: Config,
    serper_key: str,
) -> dict[str, float]:
    renderer = get_renderer(
        model_info.get_recommended_renderer_name(cfg.base_model),
        get_tokenizer(cfg.base_model),
    )
    if model_path is None:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=cfg.base_model
        )
    else:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=cfg.base_model, model_path=model_path
        )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)

    results = await asyncio.gather(
        *(eval_one(q, renderer, policy, serper_key, cfg) for q in val)
    )
    n = len(results)
    correct = sum(1 for r in results if r is True)
    extracted = sum(1 for r in results if r is not None)
    acc = correct / n if n else 0.0
    print(
        f"[{label}] accuracy={correct}/{n}={acc:.3f} | "
        f"extracted={extracted}/{n} | model_path={model_path or '(base)'}"
    )
    return {"label": label, "acc": acc, "correct": correct, "n": n, "extracted": extracted}


async def async_main(cfg: Config) -> None:
    _load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    serper_key = os.environ.get("SERPER_API_KEY")
    if not serper_key:
        raise SystemExit("SERPER_API_KEY not set.")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set.")

    val = build_validation_set(
        cfg.n_val, cfg.train_seed, cfg.train_limit, cfg.train_consumed, cfg.val_seed
    )
    print(f"Validation set: {len(val)} held-out questions (disjoint from training)\n")

    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    # Evaluate base (optional) then each checkpoint, on the SAME questions.
    labels_paths: list[tuple[str, str | None]] = []
    if cfg.include_base:
        labels_paths.append(("base", None))
    for cp in cfg.checkpoints:
        step = cp.rstrip("/").split("/")[-1]
        labels_paths.append((f"ckpt-{step}", cp))

    summary = []
    for label, path in labels_paths:
        summary.append(
            await eval_checkpoint(label, path, val, service_client, cfg, serper_key)
        )

    print("\n" + "=" * 60)
    print(f"Held-out accuracy on {len(val)} fixed questions:")
    for s in summary:
        print(f"  {s['label']:14s}  acc={s['acc']:.3f}  ({s['correct']}/{s['n']}, extracted {s['extracted']})")


def cli_main(cfg: Config) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
