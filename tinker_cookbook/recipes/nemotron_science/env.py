"""Environment for RL training on the Nemotron science open-QA dataset.

Wires nvidia/Nemotron-RL-Science-v1 (open-ended Stack Exchange science
questions) into the cookbook RL loop with:
- a safe in-process `calculator` tool (the only computation these questions
  need; see calculator.py),
- per-row answer extraction using each row's template_metadata.output_regex,
- an LLM-judge equivalence reward (grading.py), backed by a fixed Tinker-hosted
  judge model constructed lazily in make_envs() so EnvGroupBuilders stay
  pickleable.

Mirrors the structure of recipes/nemotron_mcqa/env.py.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import chz
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.nemotron_science.calculator import Calculator
from tinker_cookbook.recipes.nemotron_science.common import DEFAULT_SPLIT_DATASET
from tinker_cookbook.recipes.nemotron_science.grading import LLMJudge, extract_answer
from tinker_cookbook.renderers import Message, get_renderer, get_text_content
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollout_presets import agentic
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.tool_use import build_agent_tool_env

try:
    import weave

    _WEAVE_AVAILABLE = True
except ImportError:
    weave = None  # type: ignore[assignment]
    _WEAVE_AVAILABLE = False

_DATASET = "nvidia/Nemotron-RL-Science-v1"
_SOURCE_SPLIT = "so_openq"

SYSTEM_PROMPT = """You are an expert scientist answering an open-ended science question.

Think step by step. If a step requires numerical calculation, use the \
`calculator` tool (it evaluates one arithmetic expression, e.g. \
`sqrt(2*9.8*10)`); do not do long arithmetic in your head. When you have the \
answer, state it clearly and enclose the final answer in \\boxed{}, e.g. \
\\boxed{v \\approx 0.94c}. Keep the final answer concise."""


# ---------------------------------------------------------------------------
# Weave trace op (tags train vs val), mirrors the MCQA recipe.
# ---------------------------------------------------------------------------


def _history_to_turns(history: list[Message]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for msg in history:
        turn: dict[str, Any] = {
            "role": msg.get("role", ""),
            "content": get_text_content(msg) or "",
        }
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            turn["tool_calls"] = [
                {"name": tc.function.name, "arguments": tc.function.arguments} for tc in tool_calls
            ]
        turns.append(turn)
    return turns


def _trace_science_grade(
    question: str,
    reference: str,
    predicted: str | None,
    correct: float,
    reward: float,
    trajectory: list[dict[str, Any]],
    split: str = "train",
) -> dict[str, Any]:
    return {
        "split": split,
        "reference": reference,
        "predicted": predicted,
        "correct": correct,
        "reward": reward,
        "n_turns": len(trajectory),
    }


if _WEAVE_AVAILABLE:
    _trace_science_grade = weave.op()(_trace_science_grade)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reward: extract the answer via output_regex, judge equivalence with the LLM.
# ---------------------------------------------------------------------------


@dataclass
class ScienceJudgeReward:
    """Reward = format_coef*(has_answer - 1) + correct, where `correct` is the
    LLM judge's equivalence verdict against the reference answer."""

    question: str
    reference: str
    output_regex: str | None
    judge: LLMJudge
    format_coef: float = 0.1
    trace_weave: bool = False
    split: str = "train"

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final = get_text_content(msg)
                break
        predicted = extract_answer(final, self.output_regex) if final else None
        has_answer = float(predicted is not None)
        if predicted is None:
            correct = 0.0
        else:
            correct = float(
                await self.judge.is_equivalent(self.question, self.reference, predicted)
            )
        reward = self.format_coef * (has_answer - 1.0) + correct
        metrics = {"format": has_answer, "correct": correct}

        if self.trace_weave and _WEAVE_AVAILABLE:
            _trace_science_grade(
                question=self.question,
                reference=self.reference,
                predicted=predicted,
                correct=correct,
                reward=reward,
                trajectory=_history_to_turns(history),
                split=self.split,
            )
        return reward, metrics


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class ScienceDatum:
    question: str
    reference: str
    output_regex: str | None


def _row_to_datum(row: dict[str, Any]) -> ScienceDatum:
    tm = row.get("template_metadata")
    if isinstance(tm, str):
        tm = json.loads(tm)
    output_regex = tm.get("output_regex") if isinstance(tm, dict) else None
    return ScienceDatum(
        question=row["problem"],
        reference=row["expected_answer"],
        output_regex=output_regex,
    )


def load_science(
    limit: int | None = None,
    dataset_name: str = DEFAULT_SPLIT_DATASET,
    split: str = "train",
) -> list[ScienceDatum]:
    """Load science questions from a dataset split.

    Defaults to the persisted split repo; pass dataset_name=_DATASET with
    split=_SOURCE_SPLIT to read the upstream single-split source.
    """
    ds = cast(Dataset, load_dataset(dataset_name, split=split))
    out: list[ScienceDatum] = []
    for row in ds:
        out.append(_row_to_datum(cast("dict[str, Any]", row)))
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Env construction (shared by training builder and validation evaluator)
# ---------------------------------------------------------------------------


def _bind_effort(renderer: Renderer, effort: float) -> Renderer:
    """Bind a fixed Inkling thinking-effort onto a renderer.

    The rollout loop and the judge completer call ``build_generation_prompt``
    without an ``effort`` argument, so it would otherwise default to 0.9. We
    partial-bind the desired effort onto this renderer instance so every
    generation (policy or judge) renders the effort-conditioning system message
    the model was post-trained with. No-op for non-TMLv0 renderers that don't
    accept an ``effort`` kwarg.
    """
    orig = renderer.build_generation_prompt
    if "effort" in inspect.signature(orig).parameters:
        renderer.build_generation_prompt = partial(orig, effort=effort)  # type: ignore[method-assign]
    return renderer


# Cache judge clients per process: the judge is a fixed model, so one shared
# client per (model, renderer, effort) is reused across all envs instead of
# creating a new Tinker sampling session for every rollout (which would spawn
# thousands of sessions over a run). Keyed so different judge configs coexist.
_JUDGE_CACHE: dict[tuple[str, str | None, float], LLMJudge] = {}


def _make_judge(judge_model: str, renderer_name: str | None, effort: float) -> LLMJudge:
    key = (judge_model, renderer_name, effort)
    judge = _JUDGE_CACHE.get(key)
    if judge is None:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=judge_model)
        rname = renderer_name or model_info.get_recommended_renderer_name(judge_model)
        renderer = _bind_effort(
            get_renderer(rname, tokenizer_utils.get_tokenizer(judge_model)), effort
        )
        judge = LLMJudge(sampling_client, renderer)
        _JUDGE_CACHE[key] = judge
    return judge


def build_science_env(
    datum: ScienceDatum,
    model_name: str,
    renderer_name: str | None,
    judge_model: str,
    max_turns: int,
    max_tool_calls: int,
    max_trajectory_tokens: int,
    format_coef: float,
    trace_weave: bool,
    split: str,
    policy_effort: float = 0.3,
    judge_effort: float = 0.5,
) -> Env:
    """Build one science agent env (calculator tool + LLM-judge reward).

    The judge's sampling client is created here (inside make_envs), so
    EnvGroupBuilders stay pickleable (they hold only the judge model name).
    The policy renders at ``policy_effort`` and the judge at ``judge_effort``.
    """
    renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer: Renderer = _bind_effort(
        get_renderer(renderer_name, tokenizer_utils.get_tokenizer(model_name)),
        policy_effort,
    )
    calc = Calculator()
    tools = [calc.calculator]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[t.to_spec() for t in tools], system_prompt=SYSTEM_PROMPT
    )
    initial_messages = prefix + [Message(role="user", content=datum.question)]

    base = agentic()
    rollout_config = chz.replace(
        base,
        limits=chz.replace(
            base.limits,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_trajectory_tokens=max_trajectory_tokens,
        ),
        termination=chz.replace(base.termination, pass_all_messages_to_grader=True),
    )
    return build_agent_tool_env(
        renderer=renderer,
        tools=[calc.calculator],
        initial_messages=initial_messages,
        reward_fn=ScienceJudgeReward(
            question=datum.question,
            reference=datum.reference,
            output_regex=datum.output_regex,
            judge=_make_judge(judge_model, None, judge_effort),
            format_coef=format_coef,
            trace_weave=trace_weave,
            split=split,
        ),
        rollout_config=rollout_config,
    )


class ScienceEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of science envs sharing a question (for GRPO centering)."""

    def __init__(
        self,
        datum: ScienceDatum,
        model_name: str,
        renderer_name: str | None,
        judge_model: str,
        group_size: int,
        max_turns: int,
        max_tool_calls: int,
        max_trajectory_tokens: int,
        format_coef: float,
        trace_weave: bool = False,
        policy_effort: float = 0.3,
        judge_effort: float = 0.5,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.judge_model = judge_model
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.max_trajectory_tokens = max_trajectory_tokens
        self.format_coef = format_coef
        self.trace_weave = trace_weave
        self.policy_effort = policy_effort
        self.judge_effort = judge_effort

    async def make_envs(self) -> Sequence[Env]:
        return [
            build_science_env(
                datum=self.datum,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                judge_model=self.judge_model,
                max_turns=self.max_turns,
                max_tool_calls=self.max_tool_calls,
                max_trajectory_tokens=self.max_trajectory_tokens,
                format_coef=self.format_coef,
                trace_weave=self.trace_weave,
                split="train",
                policy_effort=self.policy_effort,
                judge_effort=self.judge_effort,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["nemotron_science"]


class ScienceRLDataset(RLDataset):
    def __init__(self, builders: list[ScienceEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return len(self.builders) // self.batch_size


@chz.chz
class ScienceDatasetBuilder(RLDatasetBuilder):
    """Builds the Nemotron science RL dataset with the calculator tool."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    judge_model: str = "thinkingmachines/Inkling"
    renderer_name: str | None = None
    dataset_name: str = DEFAULT_SPLIT_DATASET
    split: str = "train"
    max_turns: int = 8
    max_tool_calls: int = 8
    max_trajectory_tokens: int = 96 * 1024
    format_coef: float = 0.1
    seed: int = 0
    n_examples: int | None = None
    trace_weave: bool = False
    policy_effort: float = 0.3
    judge_effort: float = 0.5

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        data = load_science(limit=self.n_examples, dataset_name=self.dataset_name, split=self.split)
        rng = random.Random(self.seed)
        rng.shuffle(data)
        builders = [
            ScienceEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                judge_model=self.judge_model,
                group_size=self.group_size,
                max_turns=self.max_turns,
                max_tool_calls=self.max_tool_calls,
                max_trajectory_tokens=self.max_trajectory_tokens,
                format_coef=self.format_coef,
                trace_weave=self.trace_weave,
                policy_effort=self.policy_effort,
                judge_effort=self.judge_effort,
            )
            for datum in data
        ]
        return ScienceRLDataset(builders, self.batch_size), None


# ---------------------------------------------------------------------------
# Inline validation evaluator (runs every eval_every steps during training)
# ---------------------------------------------------------------------------


class ScienceValEvaluator(SamplingClientEvaluator):
    """Evaluate the current policy on a fixed held-out science set."""

    def __init__(
        self,
        val_set: list[ScienceDatum],
        model_name: str,
        renderer_name: str | None,
        judge_model: str,
        max_turns: int,
        max_tool_calls: int,
        max_trajectory_tokens: int,
        max_tokens: int,
        format_coef: float,
        trace_weave: bool,
        policy_effort: float = 0.3,
        judge_effort: float = 0.5,
    ):
        self.val_set = val_set
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.judge_model = judge_model
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_tokens = max_tokens
        self.format_coef = format_coef
        self.trace_weave = trace_weave
        self.policy_effort = policy_effort
        self.judge_effort = judge_effort

    async def _eval_one(
        self, datum: ScienceDatum, policy: TinkerTokenCompleter
    ) -> dict[str, float]:
        env = build_science_env(
            datum=datum,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            judge_model=self.judge_model,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            max_trajectory_tokens=self.max_trajectory_tokens,
            format_coef=self.format_coef,
            trace_weave=self.trace_weave,
            split="val",
            policy_effort=self.policy_effort,
            judge_effort=self.judge_effort,
        )
        traj = await do_single_rollout(policy, env)
        correct = 0.0
        fmt = 0.0
        total_reward = 0.0
        for t in traj.transitions:
            total_reward += t.reward
            if "correct" in t.metrics:
                correct = float(t.metrics["correct"])
            if "format" in t.metrics:
                fmt = float(t.metrics["format"])
        return {"correct": correct, "format": fmt, "reward": total_reward}

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        results = await asyncio.gather(*(self._eval_one(d, policy) for d in self.val_set))
        n = len(results) or 1
        return {
            "val/accuracy": sum(r["correct"] for r in results) / n,
            "val/format_rate": sum(r["format"] for r in results) / n,
            "val/reward": sum(r["reward"] for r in results) / n,
            "val/n": float(len(results)),
        }


@chz.chz
class ScienceValEvaluatorBuilder:
    """chz builder returning a ScienceValEvaluator (used via evaluator_builders)."""

    model_name: str
    judge_model: str = "thinkingmachines/Inkling"
    renderer_name: str | None = None
    dataset_name: str = DEFAULT_SPLIT_DATASET
    split: str = "validation"
    n_val: int | None = None
    max_turns: int = 8
    max_tool_calls: int = 8
    max_trajectory_tokens: int = 96 * 1024
    max_tokens: int = 8192
    format_coef: float = 0.1
    trace_weave: bool = False
    policy_effort: float = 0.3
    judge_effort: float = 0.5

    def __call__(self) -> ScienceValEvaluator:
        val_set = load_science(limit=self.n_val, dataset_name=self.dataset_name, split=self.split)
        return ScienceValEvaluator(
            val_set=val_set,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            judge_model=self.judge_model,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_tokens=self.max_tokens,
            format_coef=self.format_coef,
            trace_weave=self.trace_weave,
            policy_effort=self.policy_effort,
            judge_effort=self.judge_effort,
        )
