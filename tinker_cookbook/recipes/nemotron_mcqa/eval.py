"""Standalone controlled eval: score checkpoints on the held-out validation split.

Loads the persisted `validation` split (disjoint from training by construction)
and runs the same search+browse agent rollout + \\boxed{} grader against each
checkpoint, on identical questions. Apples-to-apples measurement of whether
training improved the model.

Usage:
    uv run python -m tinker_cookbook.recipes.nemotron_mcqa.eval \
        checkpoints='["tinker://.../sampler_weights/000001", "..."]' \
        provider=serper n_val=50

If checkpoints is empty, only the base model is evaluated. base_model is used
to render.
"""

from __future__ import annotations

import asyncio
import os

import chz
import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.nemotron_mcqa.common import (
    DEFAULT_SPLIT_DATASET,
    load_dotenv,
    repo_root_dotenv,
    require_provider_key,
)
from tinker_cookbook.recipes.nemotron_mcqa.env import (
    NemotronDatum,
    build_nemotron_env,
    load_nemotron,
)
from tinker_cookbook.rl.rollouts import do_single_rollout


@chz.chz
class Config:
    checkpoints: list[str] = chz.field(default_factory=list)
    base_model: str = "thinkingmachines/Inkling-Small"
    renderer_name: str | None = None
    include_base: bool = True  # also eval the untrained base model as reference
    dataset_name: str = DEFAULT_SPLIT_DATASET
    n_val: int | None = 50  # None -> whole validation split
    provider: str = "serper"
    # rollout knobs (match training defaults for an apples-to-apples comparison)
    max_turns: int = 16
    max_tool_calls: int = 30
    max_trajectory_tokens: int = 96 * 1024
    max_tokens: int = 8192
    n_results: int = 5
    browse_chars: int = 8000
    format_coef: float = 0.1
    base_url: str | None = None


async def _eval_one(
    q: NemotronDatum, policy: TinkerTokenCompleter, cfg: Config
) -> bool | None:
    """Run one rollout; return correct(bool), or None if no answer extracted."""
    env = build_nemotron_env(
        datum=q,
        model_name=cfg.base_model,
        renderer_name=cfg.renderer_name,
        max_turns=cfg.max_turns,
        max_tool_calls=cfg.max_tool_calls,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        provider=cfg.provider,
        n_results=cfg.n_results,
        browse_chars=cfg.browse_chars,
        format_coef=cfg.format_coef,
        trace_weave=False,
        split="eval",
    )
    traj = await do_single_rollout(policy, env)
    # format==0 means no \boxed{} extracted; else correct in {0,1}.
    fmt = 0.0
    correct = 0.0
    for t in traj.transitions:
        if "format" in t.metrics:
            fmt = float(t.metrics["format"])
        if "correct" in t.metrics:
            correct = float(t.metrics["correct"])
    if fmt == 0.0:
        return None
    return correct == 1.0


async def eval_checkpoint(
    label: str,
    model_path: str | None,
    val: list[NemotronDatum],
    service_client: tinker.ServiceClient,
    cfg: Config,
) -> dict[str, str | float]:
    if model_path is None:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=cfg.base_model
        )
    else:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=cfg.base_model, model_path=model_path
        )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)
    results = await asyncio.gather(*(_eval_one(q, policy, cfg) for q in val))
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
    load_dotenv(repo_root_dotenv())
    require_provider_key(cfg.provider)
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set.")

    val = load_nemotron(
        limit=cfg.n_val, dataset_name=cfg.dataset_name, split="validation"
    )
    print(f"Validation set: {len(val)} held-out questions (persisted split)\n")

    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    labels_paths: list[tuple[str, str | None]] = []
    if cfg.include_base:
        labels_paths.append(("base", None))
    for cp in cfg.checkpoints:
        step = cp.rstrip("/").split("/")[-1]
        labels_paths.append((f"ckpt-{step}", cp))

    summary = []
    for label, path in labels_paths:
        summary.append(await eval_checkpoint(label, path, val, service_client, cfg))

    print("\n" + "=" * 60)
    print(f"Held-out accuracy on {len(val)} fixed questions ({cfg.provider}):")
    for s in summary:
        print(
            f"  {s['label']:14s}  acc={s['acc']:.3f}  "
            f"({s['correct']}/{s['n']}, extracted {s['extracted']})"
        )


def cli_main(cfg: Config) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
