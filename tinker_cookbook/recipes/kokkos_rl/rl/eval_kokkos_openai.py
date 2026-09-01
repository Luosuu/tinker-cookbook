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

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HARBOR_SYSTEM_PROMPT,
    HarborTask,
    SandboxFactory,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
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
    reasoning_effort: str = "high"
    max_turns: int = 40
    max_tokens: int = 16384
    max_sampled_tokens: int = 64 * 1024
    max_tool_calls: int = 80
    sandbox_timeout: int = 3600
    command_timeout: int = 180
    grader_timeout: int = 180
    max_concurrency: int = 4
    max_infra_retries: int = 2
    task_names: str | None = None
    resume_dir: str | None = None
    contree_cache_path: str = (
        "notes/experiments/SWE-kokkos-bench/v2-pass-at-k/contree_images.json"
    )
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


def _usage(response: Any) -> tuple[int, int, int]:
    usage = response.usage
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0)
    return usage.input_tokens, usage.output_tokens, reasoning or 0


def _function_calls(response: Any) -> list[Any]:
    return [item for item in response.output if item.type == "function_call"]


async def evaluate_task(
    task: HarborTask,
    client: Any,
    sandbox_factory: SandboxFactory,
    config: CLIConfig,
    results_dir: Path,
    lock: asyncio.Lock,
) -> OpenAITaskResult:
    start = time.monotonic()
    sandbox = None
    transcript: list[dict[str, Any]] = []
    turns = tool_calls = input_tokens = output_tokens = reasoning_tokens = 0
    stop_reason = "error"
    try:
        sandbox = await sandbox_factory(task.task_dir / "environment", config.sandbox_timeout)
        bash_tool = HarborBashTool(sandbox, command_timeout=config.command_timeout)
        request_input: Any = [{"role": "user", "content": task.instruction}]
        previous_response_id: str | None = None

        while turns < config.max_turns and output_tokens < config.max_sampled_tokens:
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
            latest_input, latest_output, latest_reasoning = _usage(response)
            input_tokens = latest_input
            output_tokens += latest_output
            reasoning_tokens += latest_reasoning
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
                request_input.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": output}
                )
                transcript[-1].setdefault("tool_outputs", []).append(
                    {"call_id": call.call_id, "output": output}
                )
                tool_calls += 1
            if tool_calls >= config.max_tool_calls:
                stop_reason = "max_tool_calls"
                break
        else:
            stop_reason = "max_sampled_tokens" if output_tokens >= config.max_sampled_tokens else "max_turns"

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
                f"{result.tool_calls:>4} {result.time_seconds:>8.1f} {status:>7}\n"
            )
        (results_dir / f"{task.task_name}.json").write_text(json.dumps(transcript, indent=2))
    return result


def _load_completed(results_dir: Path) -> dict[str, OpenAITaskResult]:
    path = results_dir / "results.jsonl"
    if not path.is_file():
        return {}
    completed: dict[str, OpenAITaskResult] = {}
    for line in path.read_text().splitlines():
        result = OpenAITaskResult(**json.loads(line))
        if result.error is None:
            completed[result.task_name] = result
        else:
            completed.pop(result.task_name, None)
    return completed


async def run_eval(
    config: CLIConfig, tasks: list[HarborTask], sandbox_factory: SandboxFactory
) -> list[OpenAITaskResult]:
    from openai import AsyncOpenAI

    results_dir = (
        Path(config.resume_dir)
        if config.resume_dir
        else Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}", flush=True)
    config_dict = dump_config(config)
    if not (results_dir / "config.json").exists():
        (results_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    client = AsyncOpenAI()
    lock = asyncio.Lock()
    completed = _load_completed(results_dir)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def run_one(task: HarborTask) -> OpenAITaskResult:
        async with semaphore:
            result: OpenAITaskResult | None = None
            for attempt in range(config.max_infra_retries + 1):
                result = await evaluate_task(task, client, sandbox_factory, config, results_dir, lock)
                if result.error is None:
                    return result
                logger.warning("Retry %s (%d/%d): %s", task.task_name, attempt + 1, config.max_infra_retries, result.error)
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
        "num_passed": passed,
        "pass_at_1": passed / len(valid) if valid else None,
    }
    (results_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return results


async def main(config: CLIConfig) -> None:
    load_env_file(Path(config.env_file))
    tasks = select_tasks(load_harbor_tasks_from_dir(Path(config.tasks_dir)), config.task_names)
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
