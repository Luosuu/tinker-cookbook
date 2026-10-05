"""Run one SWE-Kokkos-Bench task with DSH on Rivanna, budget-matched to the native baseline.

The harness under test is DeepSeek Harness. Its repository access is one bash tool bridged
to the isolated Apptainer sandbox, so the model credential never enters the container. A
local proxy supplies the reference inference parameters and enforces the reference
model-request cap, because the harness itself can express neither.

Everything the run needs to be auditable is written to ``output``: the sandbox commands
and results, the resolved patch, the verifier record, and the proxy's request accounting.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class ProxyServer:
    """The comparability proxy as a managed local process."""

    def __init__(
        self,
        *,
        python: Path,
        script: Path,
        upstream: str,
        env_file: Path,
        port: int,
        max_requests: int,
        transcript: Path,
    ) -> None:
        self._python = python
        self._script = script
        self._upstream = upstream
        self._env_file = env_file
        self._port = port
        self._max_requests = max_requests
        self._transcript = transcript
        self._process: asyncio.subprocess.Process | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    async def __aenter__(self) -> ProxyServer:
        self._process = await asyncio.create_subprocess_exec(
            str(self._python),
            str(self._script),
            "--port", str(self._port),
            "--upstream", self._upstream,
            "--env-file", str(self._env_file),
            "--max-requests", str(self._max_requests),
            "--max-input-tokens", "5000000",
            "--transcript", str(self._transcript),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        for _ in range(200):
            if self._process.returncode is not None:
                output = (await self._process.stdout.read()).decode()
                raise RuntimeError(f"proxy exited early: {output}")
            try:
                _reader, writer = await asyncio.open_connection("127.0.0.1", self._port)
            except OSError:
                await asyncio.sleep(0.1)
                continue
            writer.close()
            await writer.wait_closed()
            return self
        raise RuntimeError("proxy did not become ready")

    async def __aexit__(self, *_exc: object) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()


async def run(
    *,
    task: str,
    output: Path,
    tasks_root: Path,
    runtime: Path,
    proxy_script: Path,
    env_file: Path,
    upstream: str,
    profile_patch: Path,
    python: Path,
    node: str,
    max_turns: int,
    max_tool_calls: int,
    max_tokens: int,
    wall_seconds: int,
    command_timeout: int,
    grader_timeout: int,
    sandbox_timeout: int,
) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
    from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
    from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
    from tinker_cookbook.sandbox.apptainer_sandbox import apptainer_sandbox_factory

    by_name = {t.task_name: t for t in load_harbor_tasks_from_dir(tasks_root)}
    if task not in by_name:
        raise SystemExit(f"unknown task {task}")
    harbor_task = by_name[task]

    sif = Path((harbor_task.task_dir / "environment" / "sif.path").read_text().strip())
    with sif.open("rb") as stream:
        sif_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({
        "harness": "deepseek-harness",
        "task": task,
        "sif": str(sif),
        "sif_sha256": sif_sha256,
        "model": "thinkingmachines/Inkling-Small:peft:262144",
        "upstream": upstream,
        "reference_inference": {"reasoning_effort": 0.9, "separate_reasoning": True},
        "budget": {
            "max_model_requests": max_turns,
            "max_tool_calls": max_tool_calls,
            "max_sampled_tokens": max_tokens,
            "max_wall_seconds": wall_seconds,
        },
        "network_in_sandbox": False,
    }, indent=2))

    started = time.monotonic()
    sandbox = None
    calls = 0
    reward = 0.0
    details: dict = {}
    error: str | None = None
    try:
        sandbox = await apptainer_sandbox_factory(
            harbor_task.task_dir / "environment",
            sandbox_timeout,
            allow_network=False,
            command_env={
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "CMAKE_BUILD_PARALLEL_LEVEL": "4",
                "OMP_PROC_BIND": "false",
            },
        )
        # Hidden tests, the reference solution and the credential file must all be
        # invisible from inside the container, or a score means nothing.
        isolation = await sandbox.run_command(
            "test ! -e /tests && test ! -e /solution && "
            "test ! -e /scratch/fad3ew/openhands-smoke/inkling-eval.env",
            timeout=30,
        )
        if isolation.exit_code != 0:
            raise RuntimeError("hidden assets or credentials visible in sandbox")
        (output / "preflight.json").write_text(json.dumps({
            "sif_sha256": sif_sha256,
            "hidden_assets_absent": True,
            "instruction_sha256": hashlib.sha256(harbor_task.instruction.encode()).hexdigest(),
        }, indent=2))

        with tempfile.TemporaryDirectory(prefix="dsh-bridge-", dir="/tmp") as temp:
            socket_path = str(Path(temp) / "rpc.sock")
            with (output / "events.jsonl").open("x") as events:

                async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    nonlocal calls
                    try:
                        request = json.loads(await reader.readline())
                        command = request["command"]
                        if not isinstance(command, str):
                            raise ValueError("invalid command")
                        calls += 1
                        events.write(json.dumps({"type": "bash", "index": calls, "command": command}) + "\n")
                        events.flush()
                        answer = await sandbox.run_command(
                            command, timeout=command_timeout, max_output_bytes=16384
                        )
                        response = {
                            "exit_code": answer.exit_code,
                            "stdout": answer.stdout[:16384],
                            "stderr": answer.stderr[:16384],
                        }
                        events.write(json.dumps({"type": "bash_result", "index": calls, **response}) + "\n")
                        events.flush()
                    except Exception as exc:  # noqa: BLE001 - reported to the harness
                        response = {"error": f"{type(exc).__name__}: {exc}"}
                        events.write(json.dumps({"type": "bridge_error", **response}) + "\n")
                        events.flush()
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()

                server = await asyncio.start_unix_server(handle, path=socket_path)
                os.chmod(socket_path, 0o600)
                port = _free_port()
                async with ProxyServer(
                    python=python,
                    script=proxy_script,
                    upstream=upstream,
                    env_file=env_file,
                    port=port,
                    max_requests=max_turns,
                    transcript=output / "proxy-budget.jsonl",
                ) as proxy:
                    patch_text = profile_patch.read_text().replace(
                        "${DSH_BRIDGE_PLUGIN}", str(runtime / "sandbox-bridge.mjs")
                    ).replace("${DSH_PROXY_BASE_URL}", proxy.base_url)
                    resolved_patch = output / "dsh.patch.yml"
                    resolved_patch.write_text(patch_text)

                    env = dict(os.environ)
                    env.update({
                        "DSH_HOME": str(output / "dsh-home"),
                        "HPC_BRIDGE_SOCKET": socket_path,
                        "HPC_MAX_TOOL_CALLS": str(max_tool_calls),
                        "DSH_PROXY_KEY": "proxy-local",
                        "TINKER_API_KEY": _read_key(env_file),
                    })
                    with (output / "stdout.log").open("x") as stdout, \
                         (output / "stderr.log").open("x") as stderr:
                        process = await asyncio.create_subprocess_exec(
                            node,
                            str(runtime / "node_modules/@deepseek-ai/dsh/lib/bin.js"),
                            "--profile", "headless",
                            "--patch", str(resolved_patch),
                            harbor_task.instruction,
                            cwd=str(runtime),
                            env=env,
                            stdout=stdout,
                            stderr=stderr,
                        )
                        try:
                            await asyncio.wait_for(process.wait(), timeout=wall_seconds)
                        except TimeoutError:
                            process.kill()
                            await process.wait()
                            raise RuntimeError("DSH wall-time budget exhausted") from None
                        if process.returncode != 0:
                            raise RuntimeError(
                                f"DSH exit={process.returncode}; see stderr.log"
                            )
                server.close()
                await server.wait_closed()

        # Grading happens even when the harness ended at a cap: the candidate on disk is
        # the artifact under test, exactly as in the reference harness.
        await capture_candidate(sandbox, task=harbor_task, results_dir=output)
        reward, details = await HarborReward(
            tests_dir=harbor_task.task_dir / "tests",
            sandbox=sandbox,
            grader_timeout=grader_timeout,
            raise_on_grading_error=True,
            task_name=harbor_task.task_name,
            grading_log_path=output / "grading.json",
        )([])
    except Exception as exc:  # noqa: BLE001 - recorded as the episode outcome
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            await sandbox.cleanup()

    summary = {
        "task_name": task,
        "reward": reward,
        "reward_details": details,
        "turns_used": calls,
        "time_seconds": time.monotonic() - started,
        "error": error,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _read_key(env_file: Path) -> str:
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("TINKER_API_KEY="):
            return line.split("=", 1)[1]
    raise SystemExit(f"TINKER_API_KEY not found in {env_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--proxy-script", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--profile-patch", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--node", default="node")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--command-timeout", type=int, default=900)
    parser.add_argument("--grader-timeout", type=int, default=1200)
    parser.add_argument("--sandbox-timeout", type=int, default=7200)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
