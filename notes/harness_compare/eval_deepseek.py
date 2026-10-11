"""Run one frozen SWE-Kokkos-Bench task with DeepSeek Harness on Rivanna."""

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
from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
from tinker_cookbook.sandbox.apptainer_sandbox import apptainer_sandbox_factory


async def evaluate(task_name: str, output: Path, root: Path) -> dict:
    task = next(t for t in load_harbor_tasks_from_dir(root / "tasks") if t.task_name == task_name)
    design = json.loads((root / "design.json").read_text())
    expected = design["validation"][task_name]
    image = Path((task.task_dir / "environment/sif.path").read_text().strip())
    with image.open("rb") as stream:
        image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if not all(v["sif_sha256"] == image_hash and v["baseline_ok"] for v in expected):
        raise RuntimeError("task image differs from validated snapshot")
    if {v["kind"]: v["reward"] for v in expected} != {"nop": 0.0, "oracle": 1.0}:
        raise RuntimeError("task verifier validation missing")
    output.mkdir(parents=True, exist_ok=False)
    load_env_file(root.parent / "inkling-eval.env")
    config = {
        "harness": "deepseek",
        "harness_version": "0.1.5-rc.3",
        "model": "thinkingmachines/Inkling-Small:peft:262144",
        "task": task_name,
        "source": str(root),
        "network_in_sandbox": False,
        "max_tool_calls": 80,
        "command_timeout": 900,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    sandbox = None
    calls = 0
    result = None
    try:
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
                nonlocal calls
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
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                    events.write(json.dumps({"type": "bridge_error", **response}) + "\n")
                    events.flush()
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            server = await asyncio.start_unix_server(handle, path=socket_path)
            os.chmod(socket_path, 0o600)
            runtime = Path(__file__).resolve().parent / "dsh-runtime"
            env = dict(os.environ, HPC_BRIDGE_SOCKET=socket_path,
                       DSH_HOME=str(output / "dsh-home"))
            with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
                process = await asyncio.create_subprocess_exec(
                    "node", str(runtime / "node_modules/@deepseek-ai/dsh/lib/bin.js"),
                    "--profile", "headless", "--patch", str(runtime / "deepseek.patch.yml"),
                    task.instruction, cwd=runtime, env=env, stdout=stdout, stderr=stderr,
                )
                try:
                    await asyncio.wait_for(process.wait(), timeout=3600)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    raise RuntimeError("DeepSeek Harness wall-time budget exhausted")
            server.close()
            await server.wait_closed()
            events.close()
            if process.returncode != 0:
                raise RuntimeError(f"DeepSeek Harness exit={process.returncode}; see stderr.log")
        await capture_candidate(sandbox, task=task, results_dir=output)
        reward, details = await HarborReward(
            tests_dir=task.task_dir / "tests", sandbox=sandbox, grader_timeout=900,
            raise_on_grading_error=True, task_name=task.task_name,
            grading_log_path=output / "grading.json",
        )([])
        result = TaskResult(task.task_name, 0, reward, details, calls, time.monotonic() - started)
    except Exception as exc:
        result = TaskResult(
            task.task_name, 0, 0.0, {}, calls, time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if sandbox is not None:
            await sandbox.cleanup()
    record = asdict(result)
    (output / "summary.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    root = Path(os.environ["HPC_PI_EVAL_ROOT"])
    print(json.dumps(asyncio.run(evaluate(sys.argv[1], Path(sys.argv[2]), root))))
