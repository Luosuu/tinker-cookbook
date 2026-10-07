"""Run one SWE-Kokkos-Bench task with DSH on a ConTree sandbox, budget-matched to the reference.

This is the same harness as the Rivanna runner, with the sandbox backend swapped. ConTree
removes the dependency on a cluster allocation and on prebuilt SIF images: the factory
executes the exported task Dockerfile's RUN instructions once and caches the resulting
image, then branches an isolated session per episode.

The harness process runs on this machine because it needs the model credential; every
repository command the model issues crosses the bridge into the sandbox. The local proxy
supplies the reference inference parameters and enforces the reference request and token
budgets.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
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


def _load_env_file(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


async def _run_proxy(
    *,
    python: Path,
    script: Path,
    upstream: str,
    port: int,
    max_requests: int,
    transcript: Path,
) -> asyncio.subprocess.Process:
    process = await asyncio.create_subprocess_exec(
        str(python),
        str(script),
        "--port",
        str(port),
        "--upstream",
        upstream,
        "--max-requests",
        str(max_requests),
        "--max-input-tokens",
        "5000000",
        "--transcript",
        str(transcript),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    for _ in range(200):
        if process.returncode is not None:
            raise RuntimeError(f"proxy exited early: {(await process.stdout.read()).decode()}")
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.1)
            continue
        writer.close()
        await writer.wait_closed()
        return process
    raise RuntimeError("proxy did not become ready")


def _yaml_block(text: str) -> str:
    lines = text.strip().splitlines() or [""]
    return ">-\n" + "\n".join("      " + line for line in lines)


class AffinitySandbox:
    """Confine every command on a sandbox to a fixed CPU set.

    Wraps a `SandboxInterface` so callers need no knowledge of the constraint. Grading
    goes through `HarborReward`, which calls `run_command` directly, so wrapping the
    sandbox is the only way to confine the verifier's `--parallel` build as well as the
    agent's commands.
    """

    def __init__(self, inner, cpus: str) -> None:
        self._inner = inner
        self._cpus = cpus

    def _wrap(self, command: str) -> str:
        # Non-login shell: the sandbox itself runs commands through `sh -c`, so a login
        # shell would add profile-sourcing side effects that the reference never had.
        return f"taskset -c {self._cpus} bash -c {shlex.quote(command)}"

    @property
    def sandbox_id(self) -> str:
        return self._inner.sandbox_id

    async def run_command(self, command, workdir=None, timeout=60, max_output_bytes=None):
        return await self._inner.run_command(
            self._wrap(command), workdir=workdir, timeout=timeout, max_output_bytes=max_output_bytes
        )

    async def read_file(self, path, max_bytes=None, timeout=60):
        return await self._inner.read_file(path, max_bytes=max_bytes, timeout=timeout)

    async def write_file(self, path, content, executable=False, timeout=60):
        return await self._inner.write_file(path, content, executable=executable, timeout=timeout)

    async def send_heartbeat(self, timeout: int = 30) -> None:
        await self._inner.send_heartbeat(timeout)

    async def cleanup(self) -> None:
        await self._inner.cleanup()


async def run(
    *,
    task: str,
    output: Path,
    tasks_root: Path,
    code_root: Path,
    runtime: Path,
    proxy_script: Path,
    env_file: Path,
    upstream: str,
    profile_patch: Path,
    persona_file: Path,
    python: Path,
    node: str,
    cache_path: Path,
    max_turns: int,
    max_tool_calls: int,
    nudge_after: int,
    verify_nudge: bool,
    coverage_nudge: bool,
    wall_seconds: int,
    command_timeout: int,
    grader_timeout: int,
    sandbox_timeout: int,
    build_parallelism: int,
    cpu_affinity: str | None,
    prepare_only: bool,
) -> dict:
    # Resolve before anything changes directory: the DSH launcher is spawned with cwd set
    # to the runtime, so a relative path here would be resolved twice.
    runtime = runtime.resolve()
    code_root = code_root.resolve()
    sys.path.insert(0, str(code_root))
    from tinker_cookbook.recipes.kokkos_rl.rl.tasks import prepared_kokkos_tasks

    # Load through prepared_kokkos_tasks so the verifier is the regenerated, guarded one.
    # The published payload's own test.sh grants reward 1 when its gtest filter selects
    # zero tests, which silently awards a pass for an unmodified repository. The context
    # must stay open for the whole episode because it owns the temporary task copy.
    with prepared_kokkos_tasks(tasks_root) as prepared:
        by_name = {t.task_name: t for t in prepared}
        if task not in by_name:
            raise SystemExit(f"unknown task {task}")
        harbor_task = by_name[task]
        return await _run_episode(
            task=task,
            output=output,
            harbor_task=harbor_task,
            runtime=runtime,
            proxy_script=proxy_script,
            env_file=env_file,
            upstream=upstream,
            profile_patch=profile_patch,
            persona_file=persona_file,
            python=python,
            node=node,
            cache_path=cache_path,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            wall_seconds=wall_seconds,
            command_timeout=command_timeout,
            grader_timeout=grader_timeout,
            sandbox_timeout=sandbox_timeout,
            build_parallelism=build_parallelism,
            cpu_affinity=cpu_affinity,
            nudge_after=nudge_after,
            verify_nudge=verify_nudge,
            coverage_nudge=coverage_nudge,
            prepare_only=prepare_only,
        )


async def _run_episode(
    *,
    task,
    output,
    harbor_task,
    runtime,
    proxy_script,
    env_file,
    upstream,
    profile_patch,
    persona_file,
    python,
    node,
    cache_path,
    max_turns,
    max_tool_calls,
    wall_seconds,
    command_timeout,
    grader_timeout,
    sandbox_timeout,
    build_parallelism,
    cpu_affinity,
    nudge_after,
    verify_nudge,
    coverage_nudge,
    prepare_only,
) -> dict:
    from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
    from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
    from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory

    dockerfile = harbor_task.task_dir / "environment" / "Dockerfile"
    dockerfile_sha = hashlib.sha256(dockerfile.read_bytes()).hexdigest()

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "harness": "deepseek-harness",
                "sandbox": "contree",
                "task": task,
                "dockerfile_sha256": dockerfile_sha,
                "model": "thinkingmachines/Inkling-Small:peft:262144",
                "upstream": upstream,
                "reference_inference": {"reasoning_effort": 0.9, "separate_reasoning": True},
                "budget": {
                    "max_model_requests": max_turns,
                    "max_tool_calls": max_tool_calls,
                    "max_wall_seconds": wall_seconds,
                },
                "network_in_sandbox": False,
            },
            indent=2,
        )
    )

    factory = ContreeDockerfileSandboxFactory(
        cache_path=cache_path,
        timeout=sandbox_timeout,
        runtime_build_parallelism=build_parallelism,
        allow_network=True,  # image preparation needs network to clone and build
    )

    started = time.monotonic()
    sandbox = None
    reward = 0.0
    details: dict = {}
    error: str | None = None
    harness_status = "not_started"
    calls = 0
    try:
        sandbox = await factory(harbor_task.task_dir / "environment", sandbox_timeout)
        if cpu_affinity is not None:
            sandbox = AffinitySandbox(sandbox, cpu_affinity)
        (output / "preflight.json").write_text(
            json.dumps(
                {
                    "image_prepared": True,
                    "dockerfile_sha256": dockerfile_sha,
                    "instruction_sha256": hashlib.sha256(
                        harbor_task.instruction.encode()
                    ).hexdigest(),
                },
                indent=2,
            )
        )

        if prepare_only:
            harness_status = "prepared_only"
        else:
            with tempfile.TemporaryDirectory(prefix="dsh-bridge-") as temp:
                socket_path = str(Path(temp) / "rpc.sock")
                with (output / "events.jsonl").open("x") as events:

                    async def handle(
                        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
                    ) -> None:
                        nonlocal calls
                        try:
                            request = json.loads(await reader.readline())
                            command = request["command"]
                            if not isinstance(command, str):
                                raise ValueError("invalid command")
                            calls += 1
                            events.write(
                                json.dumps({"type": "bash", "index": calls, "command": command})
                                + "\n"
                            )
                            events.flush()
                            answer = await sandbox.run_command(
                                command, timeout=command_timeout, max_output_bytes=16384
                            )
                            response = {
                                "exit_code": answer.exit_code,
                                "stdout": answer.stdout[:16384],
                                "stderr": answer.stderr[:16384],
                            }
                            events.write(
                                json.dumps({"type": "bash_result", "index": calls, **response})
                                + "\n"
                            )
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
                    port = _free_port()
                    proxy = await _run_proxy(
                        python=python,
                        script=proxy_script,
                        upstream=upstream,
                        port=port,
                        max_requests=max_turns,
                        transcript=output / "proxy-budget.jsonl",
                    )
                    try:
                        persona = persona_file.read_text().strip()
                        resolved_patch = output / "dsh.patch.yml"
                        resolved_patch.write_text(
                            profile_patch.read_text()
                            .replace("${DSH_BRIDGE_PLUGIN}", str(runtime / "sandbox-bridge.mjs"))
                            .replace("${DSH_PROXY_BASE_URL}", f"http://127.0.0.1:{port}/v1")
                            .replace("${DSH_PERSONA}", _yaml_block(persona))
                        )
                        env = dict(os.environ)
                        env.update(
                            {
                                "DSH_HOME": str(output / "dsh-home"),
                                "HPC_BRIDGE_SOCKET": socket_path,
                                "HPC_MAX_TOOL_CALLS": str(max_tool_calls),
                                # The binding budget is the request cap, and the bridge
                                # must quote it rather than the tool-call cap.
                                "HPC_MAX_REQUESTS": str(max_turns),
                                "HPC_NUDGE_AFTER": str(nudge_after),
                                "HPC_VERIFY_NUDGE": "1" if verify_nudge else "0",
                                "HPC_COVERAGE_NUDGE": "1" if coverage_nudge else "0",
                                "DSH_PROXY_KEY": "proxy-local",
                                "TINKER_API_KEY": os.environ.get("TINKER_API_KEY", "proxy-local"),
                            }
                        )
                        with (
                            (output / "stdout.log").open("x") as stdout,
                            (output / "stderr.log").open("x") as stderr,
                        ):
                            process = await asyncio.create_subprocess_exec(
                                node,
                                str(runtime / "node_modules/@deepseek-ai/dsh/lib/bin.js"),
                                "--profile",
                                "headless",
                                "--patch",
                                str(resolved_patch),
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
                                harness_status = "wall_timeout"
                            else:
                                # A non-zero exit is expected at the request cap: the proxy
                                # answers 429 and the harness reports it as terminal. The
                                # candidate on disk is still graded, matching the reference.
                                harness_status = (
                                    "finished"
                                    if process.returncode == 0
                                    else f"exit_{process.returncode}"
                                )
                    finally:
                        proxy.terminate()
                        try:
                            await asyncio.wait_for(proxy.wait(), timeout=10)
                        except TimeoutError:
                            proxy.kill()
                            await proxy.wait()
                    server.close()
                    await server.wait_closed()

            if harness_status != "prepared_only":
                await capture_candidate(sandbox, task=harbor_task, results_dir=output)
                reward, details = await HarborReward(
                    tests_dir=harbor_task.task_dir / "tests",
                    sandbox=sandbox,
                    grader_timeout=grader_timeout,
                    raise_on_grading_error=True,
                    task_name=harbor_task.task_name,
                    grading_log_path=output / "grading.json",
                )([])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            await sandbox.cleanup()

    summary = {
        "task_name": task,
        "harness_status": harness_status,
        "reward": reward,
        "reward_details": details,
        "turns_used": calls,
        "time_seconds": time.monotonic() - started,
        "error": error,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--proxy-script", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--profile-patch", type=Path, required=True)
    parser.add_argument("--persona-file", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--node", default="node")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument(
        "--nudge-after",
        type=int,
        default=0,
        help="Read-streak length that triggers a reminder on the tool result. 0 disables it, "
        "which is the default: it measured ten points WORSE on the full 100 tasks.",
    )
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--command-timeout", type=int, default=900)
    parser.add_argument("--grader-timeout", type=int, default=1200)
    parser.add_argument("--sandbox-timeout", type=int, default=3600)
    # A ConTree sandbox reports 4 CPUs but only 4 GB. The factory writes this value into
    # the sandbox environment, and the exported verifier honours it for its own build, so
    # a value above 1 makes the grader OOM (`cc1plus: Killed signal terminated`) on the
    # larger unit-test targets. Serial is slower but actually finishes; the reference ran
    # 4 only because its sandbox had 16 GB.
    parser.add_argument("--build-parallelism", type=int, default=1)
    parser.add_argument(
        "--cpu-affinity",
        default="0,1",
        help="Comma-separated CPU list to pin commands to. Needed on 4 GB sandboxes "
        "because the exported verifier builds with --parallel and ignores "
        "CMAKE_BUILD_PARALLEL_LEVEL. Measured: the full Serial1 target builds in "
        "344s on 1 CPU and 176s on 2. Four CPUs is faster on small targets but "
        "makes the verifier's own parallel build OOM (cc1plus killed), so 2 is the "
        "safe default.",
    )
    parser.add_argument(
        "--verify-nudge",
        action="store_true",
        help="Once the episode edits without compiling, remind it to build. Targets the "
        "measured bottleneck: editors that never build pass far less often.",
    )
    parser.add_argument(
        "--coverage-nudge",
        action="store_true",
        help="Remind the model when it has edited only one file late in the budget. "
        "Patches covering under half the reference fix's files pass far less often.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.env_file.is_file():
        _load_env_file(args.env_file)
    print(json.dumps(asyncio.run(run(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
