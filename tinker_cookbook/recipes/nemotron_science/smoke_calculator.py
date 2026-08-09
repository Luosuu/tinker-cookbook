"""Calculator-path smoke test for the science recipe.

The randomly-sampled dataset is ~95% conceptual, so a small random smoke batch
may never exercise the calculator tool. This script instead runs real rollouts
on a curated set of questions that genuinely require numerical calculation
(drawn from the dataset's needs-computation cases), and verifies end to end:

  1. the model actually INVOKES the `calculator` tool,
  2. an answer is extracted via the row's output format, and
  3. the LLM judge grades it.

Usage:
    uv run python -m tinker_cookbook.recipes.nemotron_science.smoke_calculator

Requires TINKER_API_KEY (loaded from repo-root .env if present).
"""

from __future__ import annotations

import asyncio
import os

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.nemotron_science.calculator import Calculator
from tinker_cookbook.recipes.nemotron_science.common import load_dotenv, repo_root_dotenv
from tinker_cookbook.recipes.nemotron_science.env import (
    SYSTEM_PROMPT,
    ScienceDatum,
    ScienceJudgeReward,
    _bind_effort,
    _make_judge,
)
from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.rl.rollout_presets import agentic
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tool_use import build_agent_tool_env

# Curated calculator-needing questions (numeric answers, from the dataset's
# needs-computation cases). The \boxed{} output_regex matches the recipe's
# default extraction.
_BOXED_REGEX = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
_CALC_QUESTIONS: list[ScienceDatum] = [
    ScienceDatum(
        question=(
            "Determine the speed (as a fraction of c) of a positron accelerated "
            "from rest through a potential difference of 1,000,000 V, using energy "
            "conservation (electron rest energy 511 keV). Give the final answer as "
            "a fraction of c in \\boxed{}."
        ),
        reference="v ≈ 0.94 c",
        output_regex=_BOXED_REGEX,
    ),
    ScienceDatum(
        question=(
            "What speed does an electron attain after being accelerated from rest "
            "through a potential difference of 1000 V? Use non-relativistic energy "
            "conservation (electron mass 9.11e-31 kg, charge 1.6e-19 C). Give the "
            "final speed in m/s in \\boxed{}."
        ),
        reference="v ≈ 1.9e7 m/s",
        output_regex=_BOXED_REGEX,
    ),
    ScienceDatum(
        question=(
            "A radioactive sample has a half-life of 8 days. What fraction of the "
            "original amount remains after 24 days? Give the fraction in \\boxed{}."
        ),
        reference="1/8",
        output_regex=_BOXED_REGEX,
    ),
]


@chz.chz
class Config:
    model_name: str = "thinkingmachines/Inkling-Small"
    judge_model: str = "thinkingmachines/Inkling"
    max_turns: int = 8
    max_tool_calls: int = 8
    max_tokens: int = 8192
    policy_effort: float = 0.3
    judge_effort: float = 0.5
    base_url: str | None = None


async def _run_one(datum: ScienceDatum, policy: TinkerTokenCompleter, cfg: Config) -> dict:
    # Build the env manually (mirroring build_science_env) so we hold the
    # Calculator instance and can assert its call_count afterwards.
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = _bind_effort(
        get_renderer(renderer_name, tokenizer_utils.get_tokenizer(cfg.model_name)),
        cfg.policy_effort,
    )
    calc = Calculator()
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[calc.calculator.to_spec()], system_prompt=SYSTEM_PROMPT
    )
    initial_messages = prefix + [Message(role="user", content=datum.question)]
    base = agentic()
    rollout_config = chz.replace(
        base,
        limits=chz.replace(
            base.limits,
            max_turns=cfg.max_turns,
            max_tool_calls=cfg.max_tool_calls,
            max_trajectory_tokens=96 * 1024,
        ),
        termination=chz.replace(base.termination, pass_all_messages_to_grader=True),
    )
    env = build_agent_tool_env(
        renderer=renderer,
        tools=[calc.calculator],
        initial_messages=initial_messages,
        reward_fn=ScienceJudgeReward(
            question=datum.question,
            reference=datum.reference,
            output_regex=datum.output_regex,
            judge=_make_judge(cfg.judge_model, None, cfg.judge_effort),
            format_coef=0.1,
        ),
        rollout_config=rollout_config,
    )
    traj = await do_single_rollout(policy, env)
    correct = 0.0
    fmt = 0.0
    for t in traj.transitions:
        if "correct" in t.metrics:
            correct = float(t.metrics["correct"])
        if "format" in t.metrics:
            fmt = float(t.metrics["format"])
    return {
        "correct": correct,
        "format": fmt,
        "calc_calls": calc.call_count,
        "n_transitions": len(traj.transitions),
    }


async def async_main(cfg: Config) -> None:
    load_dotenv(repo_root_dotenv())
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (checked env and repo-root .env).")

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    sampling_client = await service_client.create_sampling_client_async(
        base_model=cfg.model_name
    )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)

    print(f"Running {len(_CALC_QUESTIONS)} calculator-needing questions...\n")
    results = await asyncio.gather(*(_run_one(d, policy, cfg) for d in _CALC_QUESTIONS))
    n_correct = sum(r["correct"] for r in results)
    n_fmt = sum(r["format"] for r in results)
    n_used_calc = sum(1 for r in results if r["calc_calls"] > 0)
    total_calc = sum(r["calc_calls"] for r in results)
    for i, (d, r) in enumerate(zip(_CALC_QUESTIONS, results)):
        print(
            f"[Q{i}] ref={d.reference!r} calc_calls={r['calc_calls']} "
            f"correct={r['correct']} format={r['format']} turns={r['n_transitions']}"
        )
    print(f"\nused calculator:          {n_used_calc}/{len(results)}  (total calls: {total_calc})")
    print(f"format(extracted answer): {n_fmt}/{len(results)}")
    print(f"correct(judge):           {n_correct}/{len(results)}")
    if n_used_calc == 0:
        print("\nWARNING: calculator was never invoked — calculator path unverified.")


def cli_main(cfg: Config) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
