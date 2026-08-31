"""
Standalone evaluation for Harbor tasks.

Download harbor datasets:
  uvx harbor datasets download swebench-verified@1.0 -o ~/.cache/harbor/tasks/swebench-verified-1.0
  uvx harbor datasets download terminal-bench-2.0 -o ~/.cache/harbor/tasks/terminal-bench-2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import format_trajectory
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    SandboxFactory,
    _initial_messages,
    default_sandbox_factory,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.rollout_limits import RolloutLimits
from tinker_cookbook.rl.rollout_presets import RolloutConfig, agentic
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.utils.git_rev import recipe_user_metadata
from tinker_cookbook.utils.ml_log import dump_config

logger = logging.getLogger(__name__)


@chz.chz
class EvalConfig:
    """Configuration for Harbor evaluation."""

    model_name: str = "moonshotai/Kimi-K2.6"
    output_path: str = "tinker_cookbook/recipes/harbor_rl/scripts/results"
    max_turns: int = 10
    max_tokens: int = 2048
    temperature: float = 0.0
    sandbox_timeout: int = 3600
    command_timeout: int = 120
    grader_timeout: int = 60
    sandbox_build_parallelism: int | None = None
    max_tasks: int | None = None
    max_concurrency: int = 6
    checkpoint_url: str | None = None
    base_url: str | None = None
    renderer_name: str | None = None
    thinking_effort: float | None = None
    max_trajectory_tokens: int = 112 * 1024
    max_sampled_tokens: int = 64 * 1024
    max_tool_calls: int = 40
    num_samples: int = 1
    pass_at_k: str = "1"
    resume_dir: str | None = None
    max_infra_retries: int = 2


@dataclass
class TaskResult:
    task_name: str
    sample_index: int
    reward: float
    reward_details: dict[str, float]
    turns_used: int
    time_seconds: float
    error: str | None = None
    trajectory_str: str | None = None


async def evaluate_task(
    task: HarborTask,
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    sandbox_factory: SandboxFactory,
    config: EvalConfig,
    results_dir: Path,
    lock: asyncio.Lock,
    tokenizer: tokenizer_utils.Tokenizer | None = None,
    sample_index: int = 0,
) -> TaskResult:
    """Evaluate a single task: create sandbox, run agent loop, grade, cleanup.

    Writes results to files in results_dir as soon as the task completes.
    """
    start = time.monotonic()
    env_dir = task.task_dir / "environment"

    # NB: sandbox_factory is inside the try/except. If sandbox creation
    # raises (image build OOM, fishbowl timeout, etc.) the exception would
    # otherwise escape to asyncio.gather and cancel every in-flight task.
    sandbox = None
    try:
        sandbox = await sandbox_factory(env_dir, config.sandbox_timeout)
        bash_tool = HarborBashTool(sandbox, command_timeout=config.command_timeout)
        reward_fn = HarborReward(
            tests_dir=task.task_dir / "tests",
            sandbox=sandbox,
            grader_timeout=config.grader_timeout,
            raise_on_grading_error=True,
        )

        base_rollout_config = agentic()
        rollout_config = RolloutConfig(
            limits=RolloutLimits(
                max_turns=config.max_turns,
                max_trajectory_tokens=config.max_trajectory_tokens,
                max_sampled_tokens=config.max_sampled_tokens,
                max_tool_calls=config.max_tool_calls,
            ),
            parse_errors=base_rollout_config.parse_errors,
            termination=base_rollout_config.termination,
            tool_execution=base_rollout_config.tool_execution,
        )

        env = build_agent_tool_env(
            renderer=renderer,
            tools=[bash_tool.bash],
            initial_messages=_initial_messages(task, renderer, bash_tool),
            reward_fn=reward_fn,
            max_turns=config.max_turns,
            rollout_config=rollout_config,
            generation_prompt_kwargs=(
                {"effort": config.thinking_effort} if config.thinking_effort is not None else None
            ),
        )

        trajectory = await do_single_rollout(policy, env)
        reward = sum(t.reward for t in trajectory.transitions)
        reward_details = trajectory.transitions[-1].metrics if trajectory.transitions else {}
        turns_used = len(trajectory.transitions)
        elapsed = time.monotonic() - start

        trajectory_str = (
            format_trajectory(trajectory, tokenizer, only_last_transition=True)
            if tokenizer
            else None
        )

        result = TaskResult(
            task_name=task.task_name,
            sample_index=sample_index,
            reward=reward,
            reward_details=reward_details,
            turns_used=turns_used,
            time_seconds=round(elapsed, 1),
            trajectory_str=trajectory_str,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("Task %s failed: %s", task.task_name, e)
        result = TaskResult(
            task_name=task.task_name,
            sample_index=sample_index,
            reward=0.0,
            reward_details={},
            turns_used=0,
            time_seconds=round(elapsed, 1),
            error=str(e),
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning("Sandbox cleanup failed for %s: %s", task.task_name, e)

    # Write results to files immediately
    status = "ERROR" if result.error else ("PASS" if result.reward > 0 else "FAIL")
    summary_line = (
        f"{result.task_name:<40} {result.sample_index + 1:>6} {result.reward:>7.1f} "
        f"{result.turns_used:>6} "
        f"{result.time_seconds:>8.1f} {status:>7}\n"
    )

    async with lock:
        with open(results_dir / "asummary.txt", "a") as f:
            f.write(summary_line)

        if result.error:
            with open(results_dir / "aerr.txt", "a") as f:
                f.write(f"{'=' * 60}\n")
                f.write(f"Task: {result.task_name}\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"{result.error}\n\n")

        if result.trajectory_str:
            (
                results_dir / f"{result.task_name}__sample_{result.sample_index + 1:02d}.txt"
            ).write_text(result.trajectory_str)

        with open(results_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    return result


def _pass_at_k_single(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"cannot compute pass@{k} from {n} samples")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize_pass_at_k(results: list[TaskResult], k_values: list[int]) -> dict[str, object]:
    per_task: dict[str, list[TaskResult]] = {}
    for result in results:
        per_task.setdefault(result.task_name, []).append(result)

    scores: dict[str, float] = {}
    for k in k_values:
        task_scores = []
        for task_results in per_task.values():
            valid_results = [result for result in task_results if result.error is None]
            if len(valid_results) < k:
                continue
            correct = sum(result.reward > 0 for result in valid_results)
            task_scores.append(_pass_at_k_single(len(valid_results), correct, k))
        if task_scores:
            scores[str(k)] = sum(task_scores) / len(task_scores)

    return {
        "num_tasks": len(per_task),
        "num_rollouts": len(results),
        "num_valid_rollouts": sum(result.error is None for result in results),
        "num_errors": sum(result.error is not None for result in results),
        "pass_at_k": scores,
        "per_task": {
            task_name: {
                "num_samples": sum(result.error is None for result in task_results),
                "num_correct": sum(
                    result.error is None and result.reward > 0 for result in task_results
                ),
                "num_errors": sum(result.error is not None for result in task_results),
            }
            for task_name, task_results in sorted(per_task.items())
        },
    }


def _load_completed_results(results_dir: Path) -> dict[tuple[str, int], TaskResult]:
    path = results_dir / "results.jsonl"
    if not path.is_file():
        return {}
    completed: dict[tuple[str, int], TaskResult] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        result = TaskResult(**json.loads(line))
        key = (result.task_name, result.sample_index)
        if result.error is None:
            completed[key] = result
        else:
            completed.pop(key, None)
    return completed


async def _retry_infrastructure_errors(
    operation: Callable[[], Awaitable[TaskResult]],
    *,
    max_retries: int,
    task_name: str,
    sample_index: int,
) -> TaskResult:
    """Retry failed eval attempts while retaining every attempt on disk."""
    for attempt in range(max_retries + 1):
        result = await operation()
        if result.error is None:
            return result
        if attempt < max_retries:
            logger.warning(
                "Retrying task %s sample %d after infrastructure error (%d/%d): %s",
                task_name,
                sample_index + 1,
                attempt + 1,
                max_retries,
                result.error,
            )
    return result


async def run_eval(
    config: EvalConfig,
    tasks: list[HarborTask],
    sandbox_factory: SandboxFactory = default_sandbox_factory,
) -> list[TaskResult]:
    """Run evaluation on a list of Harbor tasks.

    Results are written to files in <output_path>/<timestamp>/ as each task completes.

    Args:
        config: Evaluation configuration.
        tasks: List of HarborTask to evaluate.
        sandbox_factory: Factory for creating sandboxes (defaults to Modal).

    Returns:
        List of per-task results.
    """
    if config.num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if config.max_infra_retries < 0:
        raise ValueError("max_infra_retries must be non-negative")
    k_values = sorted({int(value.strip()) for value in config.pass_at_k.split(",")})
    if not k_values or k_values[0] < 1 or k_values[-1] > config.num_samples:
        raise ValueError("pass_at_k values must be between 1 and num_samples")

    results_dir = (
        Path(config.resume_dir)
        if config.resume_dir
        else Path(config.output_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    config_dict = dump_config(config)
    config_path = results_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config_dict, indent=2))
    invocation = {
        "started_at": datetime.now().astimezone().isoformat(),
        **config_dict,
    }
    with open(results_dir / "invocations.jsonl", "a") as f:
        f.write(json.dumps(invocation) + "\n")

    lock = asyncio.Lock()

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("eval_harbor_rl"),
    )
    if config.checkpoint_url:
        sampling_client = service_client.create_sampling_client(
            model_path=config.checkpoint_url,
            base_model=config.model_name,
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    tokenizer = tokenizer_utils.get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    renderer = get_renderer(renderer_name, tokenizer)

    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    if config.max_tasks is not None:
        tasks = random.sample(tasks, min(config.max_tasks, len(tasks)))

    completed = _load_completed_results(results_dir)
    work_items = [
        (task, sample_index)
        for sample_index in range(config.num_samples)
        for task in tasks
        if (task.task_name, sample_index) not in completed
    ]
    logger.info(
        "Starting evaluation of %d rollouts (%d already complete)",
        len(work_items),
        len(completed),
    )

    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def evaluate_with_limit(task: HarborTask, sample_index: int) -> TaskResult:
        async with semaphore:

            async def operation() -> TaskResult:
                return await evaluate_task(
                    task,
                    policy,
                    renderer,
                    sandbox_factory,
                    config,
                    results_dir,
                    lock,
                    tokenizer,
                    sample_index,
                )

            return await _retry_infrastructure_errors(
                operation,
                max_retries=config.max_infra_retries,
                task_name=task.task_name,
                sample_index=sample_index,
            )

    new_results = list(
        await asyncio.gather(
            *[evaluate_with_limit(task, sample_index) for task, sample_index in work_items]
        )
    )
    task_results = list(completed.values()) + new_results
    summary = summarize_pass_at_k(task_results, k_values)
    (results_dir / "result.json").write_text(json.dumps(summary, indent=2))
    pass_at_k = summary["pass_at_k"]
    assert isinstance(pass_at_k, dict)
    print(
        "pass@k: "
        + ", ".join(
            f"pass@{key}={value:.1%}"
            for key, value in pass_at_k.items()
            if isinstance(value, float)
        )
    )
    return task_results
