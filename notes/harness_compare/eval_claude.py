"""Run one frozen SWE-Kokkos-Bench task with Claude Code on Rivanna."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
from tinker_cookbook.sandbox.apptainer_sandbox import apptainer_sandbox_factory

from completion_quality import classify_stream
from continuation import CONTINUE_PROMPT, continuation_reason, sum_usage


async def evaluate(task_name: str, output: Path, root: Path, sample_index: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "harness": "claude-code",
        "harness_version": "2.1.50",
        "model": "thinkingmachines/Inkling-Small:peft:262144",
        "task": task_name,
        "sample_index": sample_index,
        "source": str(root),
        "network_in_sandbox": False,
        "max_tool_calls": 80,
        "max_model_turns": 40,
        "command_timeout": 900,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    sandbox = None
    calls = 0
    bridge_errors = 0
    model_turns = 0
    budget_limited = False
    process_returncode = None
    completion_quality = None
    continuation_stop_reason = None
    invocation_count = 0
    stage = "snapshot"
    cleanup_error = None
    result = None
    try:
        task = next(t for t in load_harbor_tasks_from_dir(root / "tasks") if t.task_name == task_name)
        design = json.loads((root / "design.json").read_text())
        if not 0 <= sample_index < design["samples"] or task_name not in design["tasks"]:
            raise ValueError("Task or sample is outside the frozen design")
        if design["model"] != config["model"]:
            raise RuntimeError("Model differs from frozen design")
        config["max_model_turns"] = design["max_model_turns"]
        config["max_tool_calls"] = design["max_tool_calls"]
        config["max_wall_seconds"] = design["max_wall_seconds"]
        config["continuation_policy"] = design.get("continuation_policy", "none")
        config["max_continuations"] = design.get("max_continuations", 0)
        if config["continuation_policy"] not in {"none", "thinking_only_resume"}:
            raise RuntimeError("Unknown frozen continuation policy")
        if (config["max_continuations"] < 0 or
                (config["continuation_policy"] == "none" and config["max_continuations"] != 0) or
                (config["continuation_policy"] == "thinking_only_resume" and
                 config["max_continuations"] == 0)):
            raise RuntimeError("Invalid frozen continuation limit")
        (output / "config.json").write_text(json.dumps(config, indent=2))
        for filename, expected_hash in design.get("code_sha256", {}).items():
            with (Path(__file__).resolve().parent / filename).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                    raise RuntimeError(f"Evaluator code changed: {filename}")
        if "claude_package_lock_sha256" in design:
            lock = Path(__file__).resolve().parent / "claude-runtime/package-lock.json"
            with lock.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != design["claude_package_lock_sha256"]:
                    raise RuntimeError("Claude Code package lock changed since freeze")
        if "task_digest" in design and _task_digest(task) != design["task_digest"][task_name]:
            raise RuntimeError("Task payload changed since freeze")
        expected = design["validation"][task_name]
        image = Path((task.task_dir / "environment/sif.path").read_text().strip())
        image_record = design.get("image", {}).get(task_name)
        if image_record:
            stat = image.stat()
            if (str(image), stat.st_size, stat.st_mtime_ns) != (
                image_record["path"], image_record["size"], image_record["mtime_ns"]
            ):
                raise RuntimeError("SIF path, size, or modification time changed since freeze")
            image_hash = image_record["sha256"]
            if image_record["gpu"] and not os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise RuntimeError("GPU task was scheduled without CUDA_VISIBLE_DEVICES")
        else:
            with image.open("rb") as stream:
                image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if not all(v["sif_sha256"] == image_hash and v["baseline_ok"] for v in expected):
            raise RuntimeError("task image differs from validated snapshot")
        if {v["kind"]: v["reward"] for v in expected} != {"nop": 0.0, "oracle": 1.0}:
            raise RuntimeError("task verifier validation missing")
        load_env_file(root.parent / "inkling-eval.env")
        stage = "sandbox"
        sandbox = await apptainer_sandbox_factory(
            task.task_dir / "environment", 7200, allow_network=False,
            command_env={
                "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "CMAKE_BUILD_PARALLEL_LEVEL": "4", "OMP_PROC_BIND": "false",
            },
        )
        isolation = await sandbox.run_command(
            "test ! -e /scratch/fad3ew/openhands-smoke/inkling-eval.env && "
            "test ! -e /tests && test ! -e /solution", timeout=30,
        )
        if isolation.exit_code != 0:
            raise RuntimeError("hidden assets or credentials visible in sandbox")
        (output / "preflight.json").write_text(json.dumps({
            "sif_sha256": image_hash, "hidden_assets_absent": True,
            "instruction_sha256": hashlib.sha256(task.instruction.encode()).hexdigest(),
        }, indent=2))
        with tempfile.TemporaryDirectory(prefix="hpc-bridge-", dir="/tmp") as temp:
            socket_path = str(Path(temp) / "rpc.sock")
            events = (output / "events.jsonl").open("x")

            async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                nonlocal calls, bridge_errors
                try:
                    request = json.loads(await reader.readline())
                    command = request["command"]
                    if not isinstance(command, str):
                        raise ValueError("invalid command")
                    calls += 1
                    if calls > config["max_tool_calls"]:
                        raise RuntimeError("tool call budget exhausted")
                    events.write(json.dumps({"type": "bash", "index": calls, "command": command}) + "\n")
                    events.flush()
                    answer = await sandbox.run_command(command, timeout=900, max_output_bytes=16384)
                    response = {
                        "exit_code": answer.exit_code,
                        "stdout": answer.stdout[:16384],
                        "stderr": answer.stderr[:16384],
                    }
                    events.write(json.dumps({"type": "bash_result", "index": calls, **response}) + "\n")
                    events.flush()
                except Exception as exc:
                    bridge_errors += 1
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                    events.write(json.dumps({"type": "bridge_error", **response}) + "\n")
                    events.flush()
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle, path=socket_path)
            os.chmod(socket_path, 0o600)
            runtime = Path(__file__).resolve().parent / "claude-runtime"
            mcp_config = output / "mcp.json"
            mcp_config.write_text(json.dumps({"mcpServers": {"hpc": {
                "command": sys.executable,
                "args": [str(Path(__file__).resolve().parent / "mcp_hpc.py")],
                "env": {"HPC_BRIDGE_SOCKET": socket_path},
            }}}))
            env = dict(
                os.environ,
                HPC_BRIDGE_SOCKET=socket_path,
                ANTHROPIC_BASE_URL="https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api",
                ANTHROPIC_API_KEY=os.environ["TINKER_API_KEY"],
                CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
            )
            env["HOME"] = str(output / "claude-home")
            Path(env["HOME"]).mkdir()
            stage = "claude"
            cli_results = []
            tool_names = set()
            seen_messages: set[str] = set()
            session_id = None
            with (output / "stdout.log").open("x") as stdout, \
                    (output / "stderr.log").open("x") as stderr, \
                    (output / "continuations.jsonl").open("x") as continuation_log:
                while True:
                    invocation_count += 1
                    args = [
                        str(runtime / "node_modules/.bin/claude"),
                        "-p", task.instruction if session_id is None else CONTINUE_PROMPT,
                        "--model", config["model"],
                        "--tools", "",
                        "--mcp-config", str(mcp_config),
                        "--strict-mcp-config",
                        "--allowedTools=mcp__hpc__bash",
                        "--permission-mode", "dontAsk",
                        "--setting-sources", "",
                        "--disable-slash-commands",
                        "--output-format", "stream-json",
                        "--verbose",
                    ]
                    if config["continuation_policy"] == "none":
                        args.append("--no-session-persistence")
                    elif session_id is not None:
                        args.extend(("--resume", session_id))
                    process = await asyncio.create_subprocess_exec(
                        *args, cwd=runtime, env=env,
                        stdout=asyncio.subprocess.PIPE, stderr=stderr,
                        limit=8 * 1024 * 1024,
                    )
                    invocation_lines = []

                    async def record_output() -> None:
                        nonlocal model_turns, budget_limited
                        assert process.stdout is not None
                        while line := await process.stdout.readline():
                            decoded = line.decode(errors="replace")
                            invocation_lines.append(decoded)
                            stdout.write(decoded)
                            stdout.flush()
                            try:
                                event = json.loads(decoded)
                            except json.JSONDecodeError:
                                continue
                            if event.get("type") != "assistant":
                                continue
                            message_id = event.get("message", {}).get("id")
                            if message_id and message_id not in seen_messages:
                                seen_messages.add(message_id)
                                model_turns += 1
                                if model_turns > config["max_model_turns"]:
                                    budget_limited = True
                                    if process.returncode is None:
                                        try:
                                            process.terminate()
                                        except ProcessLookupError:
                                            pass

                    remaining = config["max_wall_seconds"] - (time.monotonic() - started)
                    if remaining <= 0:
                        process.kill()
                        await process.wait()
                        raise RuntimeError("Claude Code wall-time budget exhausted")
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(process.wait(), record_output()), timeout=remaining
                        )
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                        raise RuntimeError("Claude Code wall-time budget exhausted")
                    process_returncode = process.returncode
                    completion_quality = classify_stream(invocation_lines)
                    invocation_results = []
                    for line in invocation_lines:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "result":
                            invocation_results.append(event)
                        if event.get("type") == "assistant":
                            tool_names.update(
                                block.get("name")
                                for block in event.get("message", {}).get("content", [])
                                if block.get("type") == "tool_use"
                            )
                    if len(invocation_results) == 1:
                        cli_results.append(invocation_results[0])
                    continuation_log.write(json.dumps({
                        "invocation": invocation_count,
                        "session_id": (invocation_results[0].get("session_id")
                                       if invocation_results else session_id),
                        "returncode": process_returncode,
                        "completion_quality": completion_quality,
                        "model_turns_cumulative": model_turns,
                        "tool_calls_cumulative": calls,
                        "result_count": len(invocation_results),
                    }) + "\n")
                    continuation_log.flush()
                    if process_returncode != 0 and not budget_limited:
                        raise RuntimeError(f"Claude Code exit={process_returncode}; see stderr.log")
                    if not budget_limited and (
                        len(invocation_results) != 1 or invocation_results[0].get("is_error")
                    ):
                        raise RuntimeError("Claude Code did not return one successful result; see stdout.log")
                    if budget_limited:
                        continuation_stop_reason = "model_turn_budget"
                        break
                    continuation_stop_reason = continuation_reason(
                        completion_quality,
                        policy=config["continuation_policy"],
                        completed_continuations=invocation_count - 1,
                        max_continuations=config["max_continuations"],
                        model_turns=model_turns,
                        max_model_turns=config["max_model_turns"],
                        tool_calls=calls,
                        max_tool_calls=config["max_tool_calls"],
                        remaining_seconds=config["max_wall_seconds"] - (time.monotonic() - started),
                    )
                    if continuation_stop_reason != "resume":
                        break
                    new_session_id = invocation_results[0].get("session_id")
                    if not isinstance(new_session_id, str) or not new_session_id:
                        raise RuntimeError("Claude Code did not provide a resumable session ID")
                    if session_id is not None and new_session_id != session_id:
                        raise RuntimeError("Claude Code changed session ID during continuation")
                    session_id = new_session_id
            server.close()
            await server.wait_closed()
            events.close()
            if tool_names - {"mcp__hpc__bash"}:
                raise RuntimeError(f"Unexpected Claude Code tools: {sorted(tool_names)}")
            if bridge_errors:
                raise RuntimeError(f"Sandbox bridge had {bridge_errors} errors")
            if cli_results:
                (output / "claude_result.json").write_text(json.dumps({
                    "is_error": any(cli.get("is_error") for cli in cli_results),
                    "num_turns": sum(cli.get("num_turns") or 0 for cli in cli_results),
                    "usage": sum_usage(cli_results),
                    "invocations": [{
                        "session_id": cli.get("session_id"),
                        "num_turns": cli.get("num_turns"),
                        "usage": cli.get("usage"),
                        "modelUsage": cli.get("modelUsage"),
                    } for cli in cli_results],
                }, indent=2))
        stage = "candidate"
        await capture_candidate(sandbox, task=task, results_dir=output)
        stage = "grading"
        reward, details = await HarborReward(
            tests_dir=task.task_dir / "tests", sandbox=sandbox, grader_timeout=900,
            raise_on_grading_error=True, task_name=task.task_name,
            grading_log_path=output / "grading.json",
        )([])
        result = TaskResult(task.task_name, sample_index, reward, details, calls, time.monotonic() - started)
    except Exception as exc:
        result = TaskResult(
            task_name, sample_index, 0.0, {}, calls, time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox.cleanup()
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
    if cleanup_error and result.error is None:
        result = TaskResult(
            task_name, sample_index, 0.0, {}, calls, time.monotonic() - started,
            error=f"Sandbox cleanup failed: {cleanup_error}",
        )
    record = asdict(result)
    (output / "summary.json").write_text(json.dumps(record, indent=2))
    (output / "status.json").write_text(json.dumps({
        "stage": stage,
        "budget_limited": budget_limited,
        "model_turns_observed": model_turns,
        "tool_calls": calls,
        "bridge_errors": bridge_errors,
        "process_returncode": process_returncode,
        "completion_quality": completion_quality,
        "continuation_stop_reason": continuation_stop_reason,
        "invocation_count": invocation_count,
        "cleanup_error": cleanup_error,
        "error": result.error,
    }, indent=2))
    return record


if __name__ == "__main__":
    root = Path(os.environ["HPC_PI_EVAL_ROOT"])
    print(json.dumps(asyncio.run(evaluate(
        sys.argv[1], Path(sys.argv[2]), root,
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    ))))
