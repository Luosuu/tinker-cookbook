"""Pi SDK rollouts over the existing isolated Harbor sandbox and verifier."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import chz

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.harbor_rl.eval_state import prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HARBOR_SYSTEM_PROMPT,
    HarborTask,
    SandboxFactory,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.recipes.kokkos_rl.chat_inference import TINKER_CHAT_BASE_URL
from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file, select_tasks
from tinker_cookbook.recipes.kokkos_rl.rl.tasks import prepared_kokkos_tasks
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.utils.ml_log import dump_config


def pi_agent_directory(config: CLIConfig) -> Path:
    """Resolve an explicit install, environment override, or sibling source checkout."""
    configured = config.pi_agent_dir or os.environ.get("PI_HPC_AGENT_DIR")
    directory = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[4].parent / "pi-hpc-agent"
    )
    if not (directory / "runner.mjs").is_file():
        raise ValueError(
            f"Pi agent not found at {directory}; set pi_agent_dir or PI_HPC_AGENT_DIR"
        )
    return directory


@chz.chz
class CLIConfig:
    tasks_dir: str = "data/kokkos/SWE-kokkos-bench-v2"
    task_names: str | None = None
    output_path: str = "notes/experiments/kokkos_rl/pi"
    model_name: str = "thinkingmachines/Inkling-Small:peft:262144"
    base_url: str = TINKER_CHAT_BASE_URL
    api_key_env: str = "TINKER_API_KEY"
    env_file: str = ".env"
    hpc_skill: str = "hpc-debug"
    plugin_profile: str = "none"
    hpc_mode: str = "baseline"
    guidance: str = "none"
    file_tools: str = "none"
    build_check: str = "none"
    build_check_max_runs: int = 8
    build_check_max_seconds: int = 1200
    symbol_brief: str = "none"
    pi_agent_dir: str | None = None
    thinking_effort: float = 0.9
    temperature: float = 1.0
    context_window: int = 262144
    max_protocol_repairs: int = 0
    max_wall_seconds: int = 3600
    max_turns: int = 40
    max_tool_calls: int = 80
    max_tokens: int = 16384
    max_sampled_tokens: int = 65536
    max_input_tokens: int = 5_000_000
    max_context_tokens: int | None = None
    max_tool_output_chars: int = 16384
    num_samples: int = 1
    max_concurrency: int = 2
    sandbox_timeout: int = 3600
    command_timeout: int = 900
    grader_timeout: int = 900
    contree_cache_path: str = "/tmp/tinker-examples/kokkos_rl/contree_images.json"
    sandbox_build_parallelism: int = 1
    node: str = "node"


def selected_skill(config: CLIConfig) -> str:
    if config.hpc_skill not in {"hpc-debug", "swe-kokkos-debug"}:
        raise ValueError(f"Unknown hpc_skill: {config.hpc_skill}")
    return config.hpc_skill


def harness_identity(config: CLIConfig) -> dict[str, object]:
    pi_dir = pi_agent_directory(config)
    package = json.loads((pi_dir / "package.json").read_text())
    return {
        "hpc_mode": config.hpc_mode,
        "guidance": config.guidance,
        "file_tools": config.file_tools,
        "build_check": config.build_check,
        "symbol_brief": config.symbol_brief,
        "hpc_skill": selected_skill(config),
        "plugin_profile": config.plugin_profile,
        "pi_version": package["dependencies"]["@earendil-works/pi-coding-agent"],
        "agent_version": package["version"],
        "protocol_budget_support": (pi_dir / "protocol.mjs").is_file(),
        "resource_sha256": {
            name: hashlib.sha256((pi_dir / name).read_bytes()).hexdigest()
            for name in (
                "runner.mjs",
                "integration.mjs",
                "plugins.mjs",

                f"skills/{selected_skill(config)}/SKILL.md",
                "package.json",
            ) + tuple(
                name
                for name in (
                    "protocol.mjs",
                    "observation.mjs",
                    "guidance.mjs",
                    "file-tools.mjs",
                    "build-check.mjs",
                    "symbol-brief.mjs",
                    "package-lock.json",
                )
                if (pi_dir / name).is_file()
            )
        },
    }


INTERNAL_OUTPUT_CHARS = 400_000


def grader_build_command(task: HarborTask) -> str | None:
    """The visible build the task's own verifier runs, or None for non-CMake verifiers."""
    script = (task.task_dir / "tests" / "test.sh").read_text()
    targets: list[str] = []
    for target in re.findall(r"cmake --build build --target ([\w.]+)", script):
        if target not in targets:
            targets.append(target)
    if not targets:
        return None
    flags = " ".join(f"--target {target}" for target in targets)
    return f'cmake --build build {flags} --parallel "${{CMAKE_BUILD_PARALLEL_LEVEL:-1}}"'


async def sandbox_file_op(sandbox: SandboxInterface, event: dict) -> dict[str, object]:
    """Serve one read/edit file operation from the Pi agent inside the task sandbox."""
    op, path = event.get("op"), event.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        return {"ok": False, "error": f"Invalid sandbox path: {path!r}"}
    if op == "read":
        result = await sandbox.read_file(path, timeout=60)
        if result.exit_code != 0:
            return {"ok": False, "error": result.stderr.strip() or f"Cannot read {path}"}
        return {"ok": True, "content": result.stdout}
    if op == "write":
        if not isinstance(event.get("content"), str):
            return {"ok": False, "error": "write requires string content"}
        result = await sandbox.write_file(path, event["content"], timeout=120)
        return {"ok": result.exit_code == 0, "error": result.stderr.strip()}
    if op == "access":
        quoted = shlex.quote(path)
        result = await sandbox.run_command(f"test -r {quoted} && test -w {quoted}", timeout=30)
        return {"ok": result.exit_code == 0, "error": f"Cannot access {path}"}
    return {"ok": False, "error": f"Unknown file operation: {op!r}"}


async def pi_rollout(
    config: CLIConfig, task: HarborTask, sandbox: SandboxInterface, directory: Path
) -> dict[str, object]:
    """Service only explicit sandbox calls; journal events before acting on them."""
    if config.hpc_mode not in {"baseline", "prompt", "skill", "hook"}:
        raise ValueError(f"Unknown hpc_mode: {config.hpc_mode}")
    if "harness" in task.config:
        raise ValueError("Pi requires the original task; do not call apply_profile")
    pi_dir = pi_agent_directory(config)
    name = selected_skill(config)
    if config.hpc_mode == "skill":
        skill = (pi_dir / f"skills/{name}/SKILL.md").read_text()
        destination = f"/opt/hpc-harness/skills/{name}/SKILL.md"
        setup = await sandbox.run_command(f"mkdir -p /opt/hpc-harness/skills/{name}", timeout=30)
        if setup.exit_code != 0:
            raise RuntimeError("Cannot prepare sandbox skill directory")
        uploaded = await sandbox.write_file(destination, skill)
        if uploaded.exit_code != 0:
            raise RuntimeError("Cannot install HPC skill in sandbox")
    (directory / "harness.json").write_text(json.dumps(harness_identity(config), indent=2))
    payload = {
        **dump_config(config),
        "instruction": task.instruction,
        "system_prompt": HARBOR_SYSTEM_PROMPT,
        "build_check_command": grader_build_command(task) if config.build_check != "none" else None,
    }
    # Each invocation owns fresh evidence. Never overwrite an interrupted rollout.
    with (
        (directory / "events.jsonl").open("x") as events,
        (directory / "pi.stderr.log").open("x") as stderr,
    ):
        process = await asyncio.create_subprocess_exec(
            config.node,
            str(pi_dir / "runner.mjs"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            limit=16 * 1024 * 1024,
        )
        assert process.stdin is not None and process.stdout is not None
        final: dict[str, object] | None = None
        calls = 0
        try:
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
            async for raw in process.stdout:
                event = json.loads(raw)
                events.write(json.dumps(event) + "\n")
                events.flush()
                if event["type"] == "bash":
                    # Harness build checks are not model tool calls.
                    if not event.get("internal"):
                        calls += 1
                    if calls > config.max_tool_calls:
                        raise RuntimeError("Pi exceeded the sandbox tool budget")
                    command = event["command"]
                    if not isinstance(command, str):
                        raise ValueError("Invalid sandbox command")
                    # The agent summarizes harness build output itself; its first error can
                    # follow many progress lines.
                    limit = INTERNAL_OUTPUT_CHARS if event.get("internal") else (
                        config.max_tool_output_chars
                    )
                    result = await sandbox.run_command(
                        command,
                        workdir=None,
                        timeout=config.command_timeout,
                        max_output_bytes=limit,
                    )
                    reply = {
                        "id": event["id"],
                        "result": {
                            "exit_code": result.exit_code,
                            "stdout": result.stdout[:limit],
                            "stderr": result.stderr[:limit],
                        },
                    }
                    events.write(
                        json.dumps(
                            {"type": "sandbox_result", "internal": bool(event.get("internal")), **reply}
                        )
                        + "\n"
                    )
                    events.flush()
                    process.stdin.write((json.dumps(reply) + "\n").encode())
                    await process.stdin.drain()
                elif event["type"] == "file":
                    reply = {"id": event["id"], "result": await sandbox_file_op(sandbox, event)}
                    events.write(
                        json.dumps(
                            {
                                "type": "file_result",
                                "id": event["id"],
                                "op": event["op"],
                                "path": event["path"],
                                "ok": reply["result"]["ok"],
                            }
                        )
                        + "\n"
                    )
                    events.flush()
                    process.stdin.write((json.dumps(reply) + "\n").encode())
                    await process.stdin.drain()
                elif event["type"] == "result":
                    if final is not None:
                        raise RuntimeError("Duplicate Pi result")
                    final = event["result"]
                elif event["type"] != "event":
                    raise ValueError("Unknown Pi bridge event")
            if await process.wait() != 0 or final is None:
                raise RuntimeError(f"Pi rollout failed; see {directory / 'pi.stderr.log'}")
            (directory / "pi-result.json").write_text(json.dumps(final, indent=2))
            return final
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()


async def evaluate_task(
    config: CLIConfig,
    task: HarborTask,
    sample_index: int,
    factory: SandboxFactory,
    directory: Path,
) -> TaskResult:
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    sandbox = None
    turns = 0
    result: TaskResult | None = None
    try:
        sandbox = await factory(task.task_dir / "environment", config.sandbox_timeout)
        rollout = await pi_rollout(config, task, sandbox, directory)
        turns = int(str(rollout["turns"]))
        # Capture before hidden tests enter the sandbox, using the same artifact format.
        await capture_candidate(sandbox, task=task, results_dir=directory)
        reward, details = await HarborReward(
            tests_dir=task.task_dir / "tests",
            sandbox=sandbox,
            grader_timeout=config.grader_timeout,
            raise_on_grading_error=True,
            task_name=task.task_name,
            grading_log_path=directory / "grading.json",
        )([])
        details["natural_completion"] = float(rollout["stop_reason"] == "completed")
        # Report raw verifier reward separately from completion and training eligibility.
        result = TaskResult(
            task.task_name, sample_index, reward, details, turns, time.monotonic() - start
        )
    except Exception as exc:
        result = TaskResult(
            task.task_name,
            sample_index,
            0.0,
            {},
            turns,
            time.monotonic() - start,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox.cleanup()
            except Exception as exc:
                if result is None:
                    raise
                result = replace(
                    result, error=f"{result.error or ''} Cleanup failed: {exc}".strip()
                )
    assert result is not None
    (directory / "result.json").write_text(json.dumps(asdict(result), indent=2))
    return result


async def main(config: CLIConfig) -> None:
    load_env_file(Path(config.env_file))
    if not shutil.which(config.node):
        raise ValueError("Node.js is required")
    pi_dir = pi_agent_directory(config)
    preflight = await asyncio.create_subprocess_exec(
        config.node, "--input-type=module", "-e",
        "await import('@earendil-works/pi-coding-agent')",
        cwd=pi_dir,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await preflight.communicate()
    if preflight.returncode:
        raise ValueError(
            f"Pi dependencies unavailable in {pi_dir}; install the agent dependencies. "
            f"{stderr.decode(errors='replace')}"
        )
    if not os.environ.get(config.api_key_env):
        raise ValueError(f"Missing {config.api_key_env}")
    if (
        min(
            config.max_concurrency,
            config.num_samples,
            config.max_turns,
            config.max_tool_calls,
            config.max_tokens,
            config.max_sampled_tokens,
            config.max_input_tokens,
            config.max_tool_output_chars,
            config.context_window,
        )
        < 1
    ):
        raise ValueError("Concurrency, sample count and budgets must be positive")
    if not 0 <= config.thinking_effort < 1:
        raise ValueError("thinking_effort must be in [0, 1)")
    from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory

    factory = ContreeDockerfileSandboxFactory(
        cache_path=Path(config.contree_cache_path),
        timeout=config.sandbox_timeout,
        runtime_build_parallelism=config.sandbox_build_parallelism,
        allow_network=False,
    )
    directory = Path(config.output_path) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    )
    with prepared_kokkos_tasks(Path(config.tasks_dir)) as tasks:
        selected = select_tasks(tasks, config.task_names)
        if not selected:
            raise ValueError("No selected tasks")
        directory.mkdir(parents=True, exist_ok=False)
        identity = {
            **dump_config(config),
            **harness_identity(config),
        }
        prepare_eval_state(directory, identity, selected, evaluator="pi-kokkos")
        (directory / "config.json").write_text(json.dumps(identity, indent=2))
        semaphore = asyncio.Semaphore(config.max_concurrency)

        async def run_one(task: HarborTask, index: int) -> TaskResult:
            async with semaphore:
                result = await evaluate_task(
                    config,
                    task,
                    index,
                    factory,
                    directory / "rollouts" / f"{task.task_name}__{index:02d}",
                )
                with (directory / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(asdict(result)) + "\n")
                return result

        results = await asyncio.gather(
            *[run_one(task, index) for task in selected for index in range(config.num_samples)]
        )
        summary = {
            "num_samples": len(results),
            "num_errors": sum(r.error is not None for r in results),
            "num_passed": sum(r.error is None and r.reward > 0 for r in results),
            "num_natural_completions": sum(
                r.reward_details.get("natural_completion", 0) for r in results
            ),
            "eligible_for_training": False,
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Results: {directory}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(CLIConfig)))
