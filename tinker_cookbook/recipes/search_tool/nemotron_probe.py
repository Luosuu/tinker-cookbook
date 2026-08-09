"""End-to-end probe: Inkling-Small + Serper web search on Nemotron MCQA.

Verifies the training pipeline pieces WITHOUT running RL:
  1. Load N rows from nvidia/Nemotron-RL-knowledge-web_search-mcqa
     (question text lives in responses_create_params["input"], gold in
     expected_answer as a single letter).
  2. Expose a Serper.dev-backed `search` tool (no browse) via @tool.
  3. Run a real multi-turn agent rollout with Inkling-Small through
     build_agent_tool_env + do_single_rollout (effort defaults to 0.9 in the
     TMLv0 render path, matching Path-A training).
  4. Report per question: did it call search, how many times, the extracted
     \\boxed{} answer, whether extraction succeeded, and correctness.

Usage:
    uv run python -m tinker_cookbook.recipes.search_tool.nemotron_probe
    uv run python -m tinker_cookbook.recipes.search_tool.nemotron_probe \
        n_questions=5 max_turns=4 model_name=thinkingmachines/Inkling-Small

Requires SERPER_API_KEY and TINKER_API_KEY (read from repo-root .env if present).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, cast

import chz
import httpx
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import Message, get_renderer, get_text_content
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import (
    ToolResult,
    build_agent_tool_env,
    simple_tool_result,
    tool,
)

logger = logging.getLogger(__name__)

_DATASET = "nvidia/Nemotron-RL-knowledge-web_search-mcqa"
_SERPER_URL = "https://google.serper.dev/search"
_SERPER_SCRAPE_URL = "https://scrape.serper.dev"

# The dataset's own instructions ask for \boxed{LETTER}. We restate that plus
# how to use the tools. Kept short; the question already embeds options.
_SYSTEM_PROMPT = """You are an expert answering a hard multiple-choice question.

You have two tools:
- `search`: returns web result snippets (title, snippet, link) for a query.
- `browse`: returns the cleaned text of a webpage given its URL.

Typical loop: `search` for the fact you need, then `browse` a promising result \
link to read the details, then commit. Use the tools a handful of times (roughly \
2-5 total), then STOP and give your answer. You MUST end your response with the \
final option letter inside \\boxed{}, e.g. \\boxed{C}. Do not keep searching \
indefinitely; if you are unsure, give your best answer."""


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


def extract_boxed(text: str) -> str | None:
    r"""Extract the letter inside the last \boxed{...} (brace-balanced)."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    buf = []
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
    # Answer is a single option letter; tolerate "C." / "(C)" / "C"
    m = re.search(r"[A-J]", inner.upper())
    return m.group(0) if m else None


class SerperSearchTool:
    """Serper.dev tools: `search` (Google results) and `browse` (page scrape).

    Both hit Serper with the same API key: search -> google.serper.dev,
    browse -> scrape.serper.dev.
    """

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
        # Guard concurrency the way ChromaTool guards Chroma connections.
        self._sem = asyncio.Semaphore(16)
        self.call_count = 0
        self.browse_count = 0

    @tool
    async def search(
        self,
        query: Annotated[str, "A web search query."],
    ) -> ToolResult:
        """Search Google for a query and return up to 10 result snippets."""
        self.call_count += 1
        async with self._sem:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        _SERPER_URL,
                        headers={
                            "X-API-KEY": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json={"q": query, "num": self.n_results},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:  # network/provider errors -> visible to model
                return simple_tool_result(f"[search error: {e}]")

        organic = data.get("organic", [])[: self.n_results]
        if not organic:
            return simple_tool_result(f"No results for: {query}")
        lines = [f"Query: {query}"]
        for i, r in enumerate(organic, 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            lines.append(f"Result {i}: {title}\n{snippet}\n{link}")
        return simple_tool_result("\n".join(lines), metrics={"result_count": len(organic)})

    @tool
    async def browse(
        self,
        url: Annotated[str, "URL of a page to read, taken from a search result."],
    ) -> ToolResult:
        """Return the cleaned text content of a webpage."""
        self.browse_count += 1
        async with self._sem:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        _SERPER_SCRAPE_URL,
                        headers={
                            "X-API-KEY": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json={"url": url},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                return simple_tool_result(f"[browse error: {e}]")

        text = data.get("text", "") or ""
        if not text:
            return simple_tool_result(f"[no content extracted from {url}]")
        # Full pages are 40k-80k chars; truncate so multi-turn trajectories
        # don't overflow max_trajectory_tokens.
        truncated = text[: self.browse_chars]
        suffix = "" if len(text) <= self.browse_chars else "\n[... truncated ...]"
        return simple_tool_result(
            f"Content of {url}:\n{truncated}{suffix}",
            metrics={"page_chars": len(text)},
        )


def load_questions(n: int, seed: int) -> list[dict[str, Any]]:
    ds = cast(Dataset, load_dataset(_DATASET, split="train"))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        params = row["responses_create_params"]
        if isinstance(params, str):
            params = json.loads(params)
        question = params["input"]
        out.append({"question": question, "answer": row["expected_answer"].strip().upper()})
    return out


@chz.chz
class Config:
    n_questions: int = 3
    seed: int = 0
    model_name: str = "thinkingmachines/Inkling-Small"
    max_turns: int = 10
    max_tool_calls: int = 6
    max_tokens: int = 8192
    max_trajectory_tokens: int = 48 * 1024
    n_results: int = 5
    browse_chars: int = 8000
    base_url: str | None = None


async def probe_one(
    q: dict[str, Any],
    idx: int,
    renderer: Renderer,
    policy: TinkerTokenCompleter,
    serper_key: str,
    cfg: Config,
) -> dict[str, Any]:
    # Fresh tool per rollout so the counters are not shared across the
    # concurrent probes.
    serper = SerperSearchTool(
        serper_key, n_results=cfg.n_results, browse_chars=cfg.browse_chars
    )
    tools = [serper.search, serper.browse]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[t.to_spec() for t in tools],
        system_prompt=_SYSTEM_PROMPT,
    )
    initial_messages = prefix + [Message(role="user", content=q["question"])]

    captured: dict[str, Any] = {"final": None}

    async def reward_fn(history: list[Message]) -> tuple[float, dict[str, float]]:
        final = None
        for m in reversed(history):
            if m.get("role") == "assistant":
                final = get_text_content(m)
                break
        captured["final"] = final
        pred = extract_boxed(final or "")
        correct = float(pred is not None and pred == q["answer"])
        return correct, {"correct": correct, "extracted": float(pred is not None)}

    env = build_agent_tool_env(
        renderer=renderer,
        tools=tools,
        initial_messages=initial_messages,
        reward_fn=reward_fn,
        model_name=cfg.model_name,  # engages Inkling agentic + MinViableGroup defaults
        max_turns=cfg.max_turns,
        max_tool_calls=cfg.max_tool_calls,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
    )
    await do_single_rollout(policy, env)

    final = captured["final"] or ""
    pred = extract_boxed(final)
    return {
        "idx": idx,
        "gold": q["answer"],
        "pred": pred,
        "extracted": pred is not None,
        "correct": pred == q["answer"],
        "n_search": serper.call_count,
        "n_browse": serper.browse_count,
        "final_tail": final[-400:],
    }


async def async_main(cfg: Config) -> None:
    _load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    serper_key = os.environ.get("SERPER_API_KEY")
    if not serper_key:
        raise SystemExit("SERPER_API_KEY not set (checked env and repo-root .env).")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (checked env and repo-root .env).")

    logging.getLogger("httpx").setLevel(logging.WARNING)

    questions = load_questions(cfg.n_questions, cfg.seed)
    print(f"Loaded {len(questions)} questions from {_DATASET}\n")

    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    print(f"Model: {cfg.model_name} | renderer: {renderer_name}\n")

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    sampling_client = await service_client.create_sampling_client_async(
        base_model=cfg.model_name
    )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)

    # Run questions concurrently; each gets its own env + tool (single-use).
    results = await asyncio.gather(
        *(
            probe_one(q, i, renderer, policy, serper_key, cfg)
            for i, q in enumerate(questions)
        )
    )

    print("=" * 70)
    n_ext = sum(r["extracted"] for r in results)
    n_cor = sum(r["correct"] for r in results)
    n_used = sum((r["n_search"] + r["n_browse"]) > 0 for r in results)
    total_searches = sum(r["n_search"] for r in results)
    total_browses = sum(r["n_browse"] for r in results)
    for r in results:
        print(
            f"\n[Q{r['idx']}] gold={r['gold']} pred={r['pred']} "
            f"extracted={r['extracted']} correct={r['correct']} "
            f"searches={r['n_search']} browses={r['n_browse']}"
        )
        print(f"  final (tail): ...{r['final_tail']!r}")
    print("\n" + "=" * 70)
    print(f"Extraction rate : {n_ext}/{len(results)}")
    print(f"Used tools      : {n_used}/{len(results)}")
    print(f"Correct         : {n_cor}/{len(results)}")
    print(f"Total searches  : {total_searches} | Total browses: {total_browses}")


def cli_main(cfg: Config) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
