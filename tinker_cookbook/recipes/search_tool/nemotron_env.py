"""Environment for RL training Inkling on Nemotron web-search MCQA.

Wires the nvidia/Nemotron-RL-knowledge-web_search-mcqa dataset into the
cookbook RL loop with Serper-backed `search` + `browse` tools and a
\\boxed{LETTER} answer grader. Mirrors recipes/search_tool/search_env.py.

The Serper API key is read from the SERPER_API_KEY environment variable inside
make_envs(), so EnvGroupBuilder instances stay pickleable (no secrets, no live
connections stored as fields) for distributed rollout execution.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, cast

import chz
import httpx
from datasets import Dataset, load_dataset

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import Message, get_renderer, get_text_content
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollout_presets import agentic
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.tool_use import (
    ToolResult,
    build_agent_tool_env,
    simple_tool_result,
    tool,
)

# Optional Weave tracing. weave.init() is called by the training entrypoint;
# if it was never called (or weave is absent), @weave.op() is a no-op wrapper.
try:
    import weave

    _WEAVE_AVAILABLE = True
except ImportError:  # weave is an optional dependency
    weave = None  # type: ignore[assignment]
    _WEAVE_AVAILABLE = False

_DATASET = "nvidia/Nemotron-RL-knowledge-web_search-mcqa"
_SERPER_SEARCH_URL = "https://google.serper.dev/search"
_SERPER_SCRAPE_URL = "https://scrape.serper.dev"

SYSTEM_PROMPT = """You are an expert answering a hard multiple-choice question.

You have two tools:
- `search`: returns web result snippets (title, snippet, link) for a query.
- `browse`: returns the cleaned text of a webpage given its URL.

Typical loop: `search` for the fact you need, then `browse` a promising result \
link to read the details, then commit. Use the tools a handful of times (roughly \
2-5 total), then STOP and give your answer. You MUST end your response with the \
final option letter inside \\boxed{}, e.g. \\boxed{C}. Do not keep searching \
indefinitely; if you are unsure, give your best answer."""


# ---------------------------------------------------------------------------
# Answer extraction + reward
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    r"""Extract the option letter inside the last \boxed{...} (brace-balanced)."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    buf: list[str] = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    inner = "".join(buf).strip()
    m = re.search(r"[A-J]", inner.upper())
    return m.group(0) if m else None


def _history_to_turns(history: list[Message]) -> list[dict[str, Any]]:
    """Flatten a rollout history into readable {role, content, tool_calls} turns
    for Weave trace inspection."""
    turns: list[dict[str, Any]] = []
    for msg in history:
        role = msg.get("role", "")
        turn: dict[str, Any] = {"role": role, "content": get_text_content(msg) or ""}
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            turn["tool_calls"] = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in tool_calls
            ]
        turns.append(turn)
    return turns


def _grade_boxed_letter(
    question: str,
    gold_letter: str,
    final_answer: str | None,
    predicted_letter: str | None,
    reward: float,
    metrics: dict[str, float],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Identity-style grading record. Wrapped as a Weave op (when weave.init has
    run) so each rollout trajectory shows up in the Weave UI with its inputs
    (question + full conversation) and outputs (reward, correctness)."""
    return {
        "gold_letter": gold_letter,
        "predicted_letter": predicted_letter,
        "reward": reward,
        "correct": metrics.get("correct", 0.0),
        "format": metrics.get("format", 0.0),
        "n_turns": len(trajectory),
        "final_answer": final_answer,
    }


# Wrap with weave.op when available. If weave.init() was never called the op is
# inert (records nothing), so this is safe to leave decorated unconditionally.
if _WEAVE_AVAILABLE:
    _grade_boxed_letter = weave.op()(_grade_boxed_letter)  # type: ignore[misc]


@dataclass
class BoxedLetterReward:
    """Reward for MCQA: format bonus for producing \\boxed{}, +1 for the right letter.

    reward = format_coef * (has_boxed - 1) + correct
    (matches the shape of the search recipe's TextAnswerReward)

    When Weave tracing is active, each episode's full conversation is logged
    via the `_grade_boxed_letter` op so rollout trajectories appear in wandb.
    """

    gold_letter: str
    format_coef: float = 0.1
    question: str = ""
    trace_weave: bool = False

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final = get_text_content(msg)
                break
        if final is None:
            reward, metrics = -self.format_coef, {"format": 0.0, "correct": 0.0}
            pred = None
        else:
            pred = extract_boxed(final)
            has_boxed = float(pred is not None)
            correct = float(pred is not None and pred == self.gold_letter)
            reward = self.format_coef * (has_boxed - 1.0) + correct
            metrics = {"format": has_boxed, "correct": correct}

        if self.trace_weave and _WEAVE_AVAILABLE:
            _grade_boxed_letter(
                question=self.question,
                gold_letter=self.gold_letter,
                final_answer=final,
                predicted_letter=pred,
                reward=reward,
                metrics=metrics,
                trajectory=_history_to_turns(history),
            )
        return reward, metrics


# ---------------------------------------------------------------------------
# Serper tools (search + browse), constructed lazily so builders stay pickleable
# ---------------------------------------------------------------------------


class SerperTools:
    """Serper.dev `search` (google) + `browse` (scrape). Built in make_envs()."""

    def __init__(
        self,
        api_key: str,
        n_results: int = 5,
        browse_chars: int = 8000,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.n_results = n_results
        self.browse_chars = browse_chars
        self.timeout = timeout

    @tool
    async def search(
        self,
        query: Annotated[str, "A web search query."],
    ) -> ToolResult:
        """Search Google for a query and return up to 10 result snippets."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    _SERPER_SEARCH_URL,
                    headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": self.n_results},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return simple_tool_result(f"[search error: {e}]")

        organic = data.get("organic", [])[: self.n_results]
        if not organic:
            return simple_tool_result(f"No results for: {query}")
        lines = [f"Query: {query}"]
        for i, r in enumerate(organic, 1):
            lines.append(
                f"Result {i}: {r.get('title', '')}\n{r.get('snippet', '')}\n{r.get('link', '')}"
            )
        return simple_tool_result("\n".join(lines), metrics={"result_count": len(organic)})

    @tool
    async def browse(
        self,
        url: Annotated[str, "URL of a page to read, taken from a search result."],
    ) -> ToolResult:
        """Return the cleaned text content of a webpage."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    _SERPER_SCRAPE_URL,
                    headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                    json={"url": url},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return simple_tool_result(f"[browse error: {e}]")

        text = data.get("text", "") or ""
        if not text:
            return simple_tool_result(f"[no content extracted from {url}]")
        truncated = text[: self.browse_chars]
        suffix = "" if len(text) <= self.browse_chars else "\n[... truncated ...]"
        return simple_tool_result(
            f"Content of {url}:\n{truncated}{suffix}", metrics={"page_chars": len(text)}
        )


def _require_serper_key() -> str:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError(
            "SERPER_API_KEY is not set. Export it (or load it from .env) before training."
        )
    return key


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class NemotronDatum:
    question: str
    gold_letter: str


def load_nemotron(limit: int | None = None) -> list[NemotronDatum]:
    ds = cast(Dataset, load_dataset(_DATASET, split="train"))
    out: list[NemotronDatum] = []
    for row in ds:
        params = row["responses_create_params"]
        if isinstance(params, str):
            params = json.loads(params)
        out.append(
            NemotronDatum(
                question=params["input"],
                gold_letter=str(row["expected_answer"]).strip().upper(),
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Env group builder + dataset
# ---------------------------------------------------------------------------


class NemotronEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of Nemotron MCQA envs sharing a question (for GRPO centering)."""

    def __init__(
        self,
        datum: NemotronDatum,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        max_turns: int,
        max_tool_calls: int,
        max_trajectory_tokens: int,
        n_results: int,
        browse_chars: int,
        format_coef: float,
        trace_weave: bool = False,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.max_trajectory_tokens = max_trajectory_tokens
        self.n_results = n_results
        self.browse_chars = browse_chars
        self.format_coef = format_coef
        self.trace_weave = trace_weave

    async def make_envs(self) -> Sequence[Env]:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer: Renderer = get_renderer(
            renderer_name, tokenizer_utils.get_tokenizer(self.model_name)
        )
        tools_obj = SerperTools(
            _require_serper_key(),
            n_results=self.n_results,
            browse_chars=self.browse_chars,
        )
        tools = [tools_obj.search, tools_obj.browse]
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=[t.to_spec() for t in tools],
            system_prompt=SYSTEM_PROMPT,
        )
        initial_messages = prefix + [Message(role="user", content=self.datum.question)]

        # agentic() preset (Inkling default) but with a roomier turn budget.
        # pass_all_messages_to_grader=True so the reward op sees the whole
        # conversation (question + every tool turn), which is what Weave traces.
        base = agentic()
        rollout_config = chz.replace(
            base,
            limits=chz.replace(
                base.limits,
                max_turns=self.max_turns,
                max_tool_calls=self.max_tool_calls,
                max_trajectory_tokens=self.max_trajectory_tokens,
            ),
            termination=chz.replace(
                base.termination, pass_all_messages_to_grader=True
            ),
        )

        return [
            build_agent_tool_env(
                renderer=renderer,
                tools=[tools_obj.search, tools_obj.browse],
                initial_messages=initial_messages,
                reward_fn=BoxedLetterReward(
                    gold_letter=self.datum.gold_letter,
                    format_coef=self.format_coef,
                    question=self.datum.question,
                    trace_weave=self.trace_weave,
                ),
                rollout_config=rollout_config,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["nemotron_mcqa"]


class NemotronRLDataset(RLDataset):
    def __init__(self, builders: list[NemotronEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return len(self.builders) // self.batch_size


@chz.chz
class NemotronDatasetBuilder(RLDatasetBuilder):
    """Builds the Nemotron MCQA RL dataset with Serper search+browse tools."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 16
    max_tool_calls: int = 30
    max_trajectory_tokens: int = 96 * 1024
    n_results: int = 5
    browse_chars: int = 8000
    format_coef: float = 0.1
    seed: int = 0
    n_examples: int | None = None  # cap the dataset (handy for smoke tests)
    trace_weave: bool = False  # log rollout trajectories to Weave

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        data = load_nemotron(limit=self.n_examples)
        rng = random.Random(self.seed)
        rng.shuffle(data)

        builders = [
            NemotronEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                max_turns=self.max_turns,
                max_tool_calls=self.max_tool_calls,
                max_trajectory_tokens=self.max_trajectory_tokens,
                n_results=self.n_results,
                browse_chars=self.browse_chars,
                format_coef=self.format_coef,
                trace_weave=self.trace_weave,
            )
            for datum in data
        ]
        return NemotronRLDataset(builders, self.batch_size), None
