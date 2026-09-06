"""Evaluate an OpenAI Responses API model on exported Kokkos Harbor tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chz

from tinker_cookbook.recipes.harbor_rl.eval_state import prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HARBOR_SYSTEM_PROMPT,
    HarborTask,
    SandboxFactory,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.recipes.kokkos_rl.rl.tasks import prepared_kokkos_tasks
from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory
from tinker_cookbook.tool_use import ToolInput
from tinker_cookbook.utils.ml_log import dump_config

logger = logging.getLogger(__name__)

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "bash",
    "description": "Execute a bash command in the sandbox environment.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}


@chz.chz
class CLIConfig:
    model_name: str = "gpt-5.6-terra"
    tasks_dir: str = "data/kokkos/SWE-kokkos-bench-v2"
    output_path: str = "notes/experiments/SWE-kokkos-bench/gpt-5.6-terra/pass-at-1"
    env_file: str = ".env"
    reasoning_effort: str = "medium"
    max_turns: int = 24
    max_tokens: int = 8192
    max_sampled_tokens: int = 32 * 1024
    max_input_tokens: int = 750_000
    max_tool_calls: int = 48
    max_tool_output_chars: int = 12_000
    max_cost_usd_per_task: float | None = 0.75
    input_price_per_million: float = 2.0
    cached_input_price_per_million: float = 0.2
    cache_write_price_per_million: float = 2.5
    output_price_per_million: float = 12.0
    sandbox_timeout: int = 3600
    # Kokkos' serial unit-test targets take about 400 seconds on ConTree even
    # without competing rollouts. Leave headroom for backend load variance.
    command_timeout: int = 900
    grader_timeout: int = 180
    max_concurrency: int = 2
    max_infra_retries: int = 2
    task_names: str | None = None
    resume_dir: str | None = None
    contree_cache_path: str = "notes/experiments/SWE-kokkos-bench/v2-pass-at-k/contree_images.json"
    allow_network: bool = False


@dataclass
class OpenAITaskResult:
    task_name: str
    reward: float
    reward_details: dict[str, float]
    turns_used: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    time_seconds: float
    stop_reason: str
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: str | None = None


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def select_tasks(tasks: list[HarborTask], task_names: str | None) -> list[HarborTask]:
    if task_names is None:
        return tasks
    ordered_names = [name.strip() for name in task_names.split(",") if name.strip()]
    by_name = {task.task_name: task for task in tasks}
    missing = [name for name in ordered_names if name not in by_name]
    if missing:
        raise ValueError(f"unknown task names: {missing}")
    return [by_name[name] for name in ordered_names]


def _usage(response: Any) -> tuple[int, int, int, int, int]:
    usage = response.usage
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0)
    input_details = getattr(usage, "input_tokens_details", None)
    cached = getattr(input_details, "cached_tokens", 0) or 0
    cache_write = getattr(input_details, "cache_write_tokens", 0) or 0
    return usage.input_tokens, usage.output_tokens, reasoning or 0, cached, cache_write


def _request_cost_usd(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_input_tokens: int,
    output_tokens: int,
    config: CLIConfig,
) -> float:
    cached = min(input_tokens, cached_input_tokens)
    cache_write = min(input_tokens - cached, cache_write_input_tokens)
    uncached = input_tokens - cached - cache_write
    return (
        uncached * config.input_price_per_million
        + cached * config.cached_input_price_per_million
        + cache_write * config.cache_write_price_per_million
        + output_tokens * config.output_price_per_million
    ) / 1_000_000


def _truncate_tool_output(output: str, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("max_tool_output_chars must be positive")
    if len(output) <= max_chars:
        return output
    marker = f"\n... {len(output) - max_chars:,} characters omitted ...\n"
    available = max_chars - len(marker)
    if available <= 0:
        return output[-max_chars:]
    head_chars = available // 3
    tail_chars = available - head_chars
    return output[:head_chars] + marker + output[-tail_chars:]


def _function_calls(response: Any) -> list[Any]:
    return [item for item in response.output if item.type == "function_call"]


async def evaluate_task(
    task: HarborTask,
    client: Any,
    sandbox_factory: SandboxFactory,
    config: CLIConfig,
    results_dir: Path,
    lock: asyncio.Lock,
    prior_cost_usd: float = 0.0,
) -> OpenAITaskResult:
    start = time.monotonic()
    sandbox = None
    transcript: list[dict[str, Any]] = []
    turns = tool_calls = input_tokens = output_tokens = reasoning_tokens = 0
    cached_input_tokens = cache_write_input_tokens = 0
    estimated_cost_usd = 0.0
    stop_reason = "error"
    try:
        sandbox = await sandbox_factory(task.task_dir / "environment", config.sandbox_timeout)
        bash_tool = HarborBashTool(sandbox, command_timeout=config.command_timeout)
        request_input: Any = [{"role": "user", "content": task.instruction}]
        previous_response_id: str | None = None

        while (
            turns < config.max_turns
            and output_tokens < config.max_sampled_tokens
            and input_tokens < config.max_input_tokens
            and (
                config.max_cost_usd_per_task is None
                or prior_cost_usd + estimated_cost_usd < config.max_cost_usd_per_task
            )
        ):
            remaining = max(16, min(config.max_tokens, config.max_sampled_tokens - output_tokens))
            kwargs: dict[str, Any] = {
                "model": config.model_name,
                "instructions": HARBOR_SYSTEM_PROMPT,
                "input": request_input,
                "tools": [BASH_TOOL],
                "reasoning": {"effort": config.reasoning_effort},
                "max_output_tokens": remaining,
            }
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id
            response = await client.responses.create(**kwargs)
            turns += 1
            previous_response_id = response.id
            (
                latest_input,
                latest_output,
                latest_reasoning,
                latest_cached_input,
                latest_cache_write_input,
            ) = _usage(response)
            input_tokens += latest_input
            output_tokens += latest_output
            reasoning_tokens += latest_reasoning
            cached_input_tokens += latest_cached_input
            cache_write_input_tokens += latest_cache_write_input
            estimated_cost_usd += _request_cost_usd(
                input_tokens=latest_input,
                cached_input_tokens=latest_cached_input,
                cache_write_input_tokens=latest_cache_write_input,
                output_tokens=latest_output,
                config=config,
            )
            calls = _function_calls(response)
            transcript.append(
                {
                    "turn": turns,
                    "response_id": response.id,
                    "output_text": response.output_text,
                    "usage": {
                        "input_tokens": latest_input,
                        "output_tokens": latest_output,
                        "reasoning_tokens": latest_reasoning,
                        "cached_input_tokens": latest_cached_input,
                        "cache_write_input_tokens": latest_cache_write_input,
                        "estimated_cost_usd": round(estimated_cost_usd, 6),
                    },
                    "function_calls": [
                        {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                        for call in calls
                    ],
                }
            )
            if not calls:
                stop_reason = "model_finished"
                break

            remaining_calls = config.max_tool_calls - tool_calls
            if remaining_calls <= 0:
                stop_reason = "max_tool_calls"
                break
            calls = calls[:remaining_calls]
            request_input = []
            for call in calls:
                if call.name != "bash":
                    output = json.dumps({"error": f"unknown tool: {call.name}"})
                else:
                    arguments = json.loads(call.arguments)
                    result = await bash_tool.bash.run(
                        ToolInput(arguments=arguments, call_id=call.call_id)
                    )
                    output = str(result.messages[0]["content"])
                model_output = _truncate_tool_output(output, config.max_tool_output_chars)
                request_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": model_output,
                    }
                )
                transcript[-1].setdefault("tool_outputs", []).append(
                    {
                        "call_id": call.call_id,
                        "output": output,
                        "model_output_truncated": model_output != output,
                    }
                )
                tool_calls += 1
            if tool_calls >= config.max_tool_calls:
                stop_reason = "max_tool_calls"
                break
        else:
            if output_tokens >= config.max_sampled_tokens:
                stop_reason = "max_sampled_tokens"
            elif input_tokens >= config.max_input_tokens:
                stop_reason = "max_input_tokens"
            elif (
                config.max_cost_usd_per_task is not None
                and prior_cost_usd + estimated_cost_usd >= config.max_cost_usd_per_task
            ):
                stop_reason = "max_cost_usd"
            else:
                stop_reason = "max_turns"

        reward, reward_details = await HarborReward(
            tests_dir=task.task_dir / "tests",
            sandbox=sandbox,
            grader_timeout=config.grader_timeout,
            raise_on_grading_error=True,
            task_name=task.task_name,
        )([])
        result = OpenAITaskResult(
            task_name=task.task_name,
            reward=reward,
            reward_details=reward_details,
            turns_used=turns,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            time_seconds=round(time.monotonic() - start, 1),
            stop_reason=stop_reason,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            estimated_cost_usd=round(estimated_cost_usd, 6),
        )
    except Exception as error:
        logger.exception("Task %s failed", task.task_name)
        result = OpenAITaskResult(
            task_name=task.task_name,
            reward=0.0,
            reward_details={},
            turns_used=turns,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            time_seconds=round(time.monotonic() - start, 1),
            stop_reason="error",
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            estimated_cost_usd=round(estimated_cost_usd, 6),
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox.cleanup()
            except Exception:
                logger.warning("Sandbox cleanup failed for %s", task.task_name, exc_info=True)

    status = "ERROR" if result.error else ("PASS" if result.reward > 0 else "FAIL")
    async with lock:
        with (results_dir / "results.jsonl").open("a") as file:
            file.write(json.dumps(asdict(result)) + "\n")
        with (results_dir / "asummary.txt").open("a") as file:
            file.write(
                f"{result.task_name:<40} {result.reward:>5.1f} {result.turns_used:>4} "
                f"{result.tool_calls:>4} ${result.estimated_cost_usd:>7.3f} "
                f"{result.time_seconds:>8.1f} {status:>7}\n"
            )
        (results_dir / f"{task.task_name}.json").write_text(json.dumps(transcript, indent=2))
    return result


def _load_attempts(results_dir: Path) -> list[OpenAITaskResult]:
    path = results_dir / "results.jsonl"
    if not path.is_file():
        return []
    return [
        OpenAITaskResult(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _load_completed(results_dir: Path) -> dict[str, OpenAITaskResult]:
    completed: dict[str, OpenAITaskResult] = {}
    for result in _load_attempts(results_dir):
        if result.error is None:
            completed[result.task_name] = result
        else:
            completed.pop(result.task_name, None)
    return completed


async def run_eval(
    config: CLIConfig, tasks: list[HarborTask], sandbox_factory: SandboxFactory
) -> list[OpenAITaskResult]:
    from openai import AsyncOpenAI

    if config.max_concurrency < 1 or config.max_infra_retries < 0:
        raise ValueError("max_concurrency must be positive and max_infra_retries non-negative")
    if config.max_cost_usd_per_task is not None and config.max_cost_usd_per_task <= 0:
        raise ValueError("max_cost_usd_per_task must be positive or None")
    results_dir = (
        Path(config.resume_dir)
        if config.resume_dir
        else Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}", flush=True)
    config_dict = dump_config(config)
    prepare_eval_state(results_dir, config_dict, tasks, evaluator="openai-kokkos")
    if not (results_dir / "config.json").exists():
        (results_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    client = AsyncOpenAI()
    lock = asyncio.Lock()
    task_names = {task.task_name for task in tasks}
    attempts = [r for r in _load_attempts(results_dir) if r.task_name in task_names]
    completed = {
        name: result for name, result in _load_completed(results_dir).items() if name in task_names
    }
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def run_one(task: HarborTask) -> OpenAITaskResult:
        async with semaphore:
            previous = [r for r in attempts if r.task_name == task.task_name]
            spent = sum(r.estimated_cost_usd for r in previous)
            result: OpenAITaskResult | None = previous[-1] if previous else None
            for attempt in range(config.max_infra_retries + 1):
                if (
                    config.max_cost_usd_per_task is not None
                    and spent >= config.max_cost_usd_per_task
                ):
                    if result is None:
                        raise ValueError("max_cost_usd_per_task must be positive or None")
                    logger.warning("Task %s exhausted its cumulative cost budget", task.task_name)
                    return result
                result = await evaluate_task(
                    task,
                    client,
                    sandbox_factory,
                    config,
                    results_dir,
                    lock,
                    prior_cost_usd=spent,
                )
                attempts.append(result)
                spent += result.estimated_cost_usd
                if result.error is None:
                    return result
                logger.warning(
                    "Retry %s (%d/%d): %s",
                    task.task_name,
                    attempt + 1,
                    config.max_infra_retries,
                    result.error,
                )
            assert result is not None
            return result

    new_results = await asyncio.gather(
        *[run_one(task) for task in tasks if task.task_name not in completed]
    )
    results = list(completed.values()) + list(new_results)
    valid = [result for result in results if result.error is None]
    passed = sum(result.reward > 0 for result in valid)
    summary = {
        "model": config.model_name,
        "num_tasks": len(results),
        "num_valid": len(valid),
        "num_errors": len(results) - len(valid),
        "num_attempts": len(attempts),
        "num_passed": passed,
        "pass_at_1": passed / len(valid) if valid else None,
        "input_tokens": sum(result.input_tokens for result in attempts),
        "cached_input_tokens": sum(result.cached_input_tokens for result in attempts),
        "cache_write_input_tokens": sum(result.cache_write_input_tokens for result in attempts),
        "output_tokens": sum(result.output_tokens for result in attempts),
        "reasoning_tokens": sum(result.reasoning_tokens for result in attempts),
        "estimated_cost_usd": round(sum(result.estimated_cost_usd for result in attempts), 6),
    }
    (results_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return results


async def main(config: CLIConfig) -> None:
    load_env_file(Path(config.env_file))
    tasks_dir = Path(config.tasks_dir)
    with prepared_kokkos_tasks(tasks_dir) as prepared:
        tasks = select_tasks(prepared, config.task_names)
        sandbox_factory = ContreeDockerfileSandboxFactory(
            cache_path=Path(config.contree_cache_path),
            timeout=config.sandbox_timeout,
            allow_network=config.allow_network,
        )
        print(
            f"Running {len(tasks)} tasks with {config.model_name}, reasoning={config.reasoning_effort}",
            flush=True,
        )
        await run_eval(config, tasks, sandbox_factory)


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(CLIConfig)))
