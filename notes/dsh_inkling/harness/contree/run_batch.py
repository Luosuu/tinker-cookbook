"""Run a DSH episode set on ConTree with bounded concurrency.

One process per episode keeps a crash, a hung sandbox, or a runaway agent from taking down
the rest of the run, and lets the parallelism match ConTree's per-instance limit rather than
a single process's event loop.

Usage::

    python -m notes.dsh_inkling.harness.contree.run_batch \
        --tasks-file tasks.txt --run-dir /tmp/dsh-contree/run01 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


async def run_one(
    task: str,
    *,
    run_dir: Path,
    repo: Path,
    tasks_root: Path,
    runtime: Path,
    proxy: Path,
    env_file: Path,
    upstream: str,
    profile: Path,
    persona: Path,
    cache: Path,
    concurrency_slot: int,
    timeout: float,
    prepare_only: bool,
    cpu_affinity: str | None,
    nudge_after: int,
) -> dict:
    output = run_dir / "results" / task
    command = [
        sys.executable,
        str(repo / "notes/dsh_inkling/harness/contree/run_task_contree.py"),
        task,
        str(output),
        "--tasks-root",
        str(tasks_root),
        "--code-root",
        str(repo),
        "--runtime",
        str(runtime),
        "--proxy-script",
        str(proxy),
        "--env-file",
        str(env_file),
        "--upstream",
        upstream,
        "--profile-patch",
        str(profile),
        "--persona-file",
        str(persona),
        "--cache-path",
        str(cache),
    ]
    if prepare_only:
        command.append("--prepare-only")
    if cpu_affinity is not None:
        command += ["--cpu-affinity", str(cpu_affinity)]
    command += ["--nudge-after", str(nudge_after)]
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(repo),
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        code = process.returncode
    except TimeoutError:
        process.kill()
        await process.communicate()
        stdout, code = b"", -1
    summary_path = output / "summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.is_file()
        else {
            "task_name": task,
            "harness_status": "driver_timeout" if code == -1 else f"driver_exit_{code}",
            "reward": 0.0,
            "turns_used": 0,
            "time_seconds": time.monotonic() - started,
            "error": "driver produced no summary",
        }
    )
    summary["_slot"] = concurrency_slot
    summary["_driver_output_tail"] = (
        stdout.decode(errors="replace")[-2000:] if code not in (0, None) else ""
    )
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks-root", type=Path, default=REPO_ROOT / "data/kokkos/SWE-kokkos-bench-v2"
    )
    parser.add_argument("--runtime", type=Path, default=REPO_ROOT / ".dsh-runtime")
    parser.add_argument(
        "--proxy", type=Path, default=REPO_ROOT / "notes/dsh_inkling/harness/contree/proxy.py"
    )
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "notes/dsh_inkling/harness/contree/dsh.patch.yml.in",
    )
    parser.add_argument(
        "--persona",
        type=Path,
        default=REPO_ROOT / "notes/dsh_inkling/harness/contree/persona-reference.txt",
    )
    parser.add_argument("--cache", type=Path, default=Path("/tmp/dsh-contree/cache/images.json"))
    parser.add_argument(
        "--upstream",
        default="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--episode-timeout", type=float, default=5400)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--nudge-after",
        type=int,
        default=0,
        help="Read-streak reminder threshold; 0 is off and is the default.",
    )
    parser.add_argument(
        "--cpu-affinity",
        type=int,
        default=0,
        help="Pin each episode's commands to one CPU. Required on 4 GB sandboxes because "
        "the exported verifier builds with --parallel and ignores "
        "CMAKE_BUILD_PARALLEL_LEVEL. Pass -1 to disable.",
    )
    args = parser.parse_args()

    tasks = [line.strip() for line in args.tasks_file.read_text().splitlines() if line.strip()]
    if not tasks:
        raise SystemExit(f"no tasks in {args.tasks_file}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "results").mkdir(exist_ok=True)
    (args.run_dir / "tasks.txt").write_text("\n".join(tasks) + "\n")

    slots = asyncio.Semaphore(args.concurrency)
    print(
        json.dumps({"event": "start", "tasks": len(tasks), "concurrency": args.concurrency}),
        flush=True,
    )
    started = time.monotonic()

    async def guarded(index: int, task: str) -> tuple[int, dict]:
        async with slots:
            summary = await run_one(
                task,
                run_dir=args.run_dir,
                repo=REPO_ROOT,
                tasks_root=args.tasks_root,
                runtime=args.runtime,
                proxy=args.proxy,
                env_file=args.env_file,
                upstream=args.upstream,
                profile=args.profile,
                persona=args.persona,
                cache=args.cache,
                concurrency_slot=index % args.concurrency,
                timeout=args.episode_timeout,
                prepare_only=args.prepare_only,
                cpu_affinity=None if args.cpu_affinity == "none" else args.cpu_affinity,
                nudge_after=args.nudge_after,
            )
            return index, summary

    results = await asyncio.gather(*(guarded(i, t) for i, t in enumerate(tasks)))
    ordered = [summary for _, summary in sorted(results, key=lambda pair: pair[0])]
    (args.run_dir / "batch-summary.json").write_text(
        json.dumps(
            {
                "tasks": len(tasks),
                "concurrency": args.concurrency,
                "elapsed_seconds": time.monotonic() - started,
                "solved": sum(1 for s in ordered if s.get("reward") == 1.0),
                "episodes": ordered,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "event": "done",
                "tasks": len(tasks),
                "solved": sum(1 for s in ordered if s.get("reward") == 1.0),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
