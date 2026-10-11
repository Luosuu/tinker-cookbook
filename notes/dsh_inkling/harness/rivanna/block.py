"""Plan, freeze, run and summarize a same-period block on Rivanna.

A block runs every arm (native plus DSH variants), several draws each, over the same tasks
in one Slurm array, with arm order interleaved so no arm gets a head start: scores drift
between sessions by several points, so arms are only comparable within one block. Each
array element runs one episode in its own 4-CPU/16 GB allocation, so there is no sandbox
contention to control for and no image cache to warm or lose.

Usage (on Rivanna, from the block's pinned code checkout)::

    python -m notes.dsh_inkling.harness.rivanna.block plan BLOCK --tasks-root TASKS \\
        --arm native=native --arm control= --arm state=--state-target --draws 4
    python -m notes.dsh_inkling.harness.rivanna.block freeze BLOCK   # under Slurm
    bash notes/dsh_inkling/harness/rivanna/launch_block.sh BLOCK 36 2
    python -m notes.dsh_inkling.harness.rivanna.block status BLOCK

Arm runs land in ``BLOCK/runs/<arm>-d<draw>/results/<task>/summary.json``, the layout that
``compare_runs`` and ``block_report`` read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
HARNESS = REPO_ROOT / "notes/dsh_inkling/harness"
UPSTREAM = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# The sandbox must outlive a full 3600 s episode plus a 2400 s verifier build.
EPISODE_BUDGET = {
    "max_model_requests": 40,
    "max_tool_calls": 80,
    "wall_seconds": 3600,
    "command_timeout": 900,
    "grader_timeout": 2400,
    "sandbox_timeout": 7200,
    "native_max_infra_retries": 4,
}


def parse_arm(spec: str) -> tuple[str, dict]:
    name, _, flags = spec.partition("=")
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"invalid arm name in {spec!r}")
    if flags.strip() == "native":
        return name, {"kind": "native", "flags": []}
    return name, {"kind": "dsh", "flags": shlex.split(flags)}


def interleave(tasks: list[str], arms: list[str], draws: int, seed: int) -> list[dict]:
    """Order episodes draw by draw, task by task, cycling through every arm.

    Each draw shuffles the task order independently; within a task all arms are adjacent,
    so at any point in the queue every arm has started the same number of episodes, plus
    or minus one.
    """
    rng = random.Random(seed)
    slots: list[dict] = []
    for draw in range(draws):
        order = list(tasks)
        rng.shuffle(order)
        for task in order:
            arm_order = list(arms)
            rng.shuffle(arm_order)
            for arm in arm_order:
                slots.append({"arm": arm, "draw": draw, "run": f"{arm}-d{draw}", "task": task})
    return slots


def git_identity(repo: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan(args: argparse.Namespace) -> None:
    from notes.dsh_inkling.harness.rivanna.apptainer import (
        require_guarded_verifier,
        requires_gpu,
        sif_path,
    )

    block: Path = args.block.resolve()
    if (block / "design.json").exists():
        raise SystemExit(f"refusing: {block}/design.json exists")
    identity = git_identity(REPO_ROOT)
    if identity["dirty"] and not args.allow_dirty:
        raise SystemExit("refusing: the code checkout has uncommitted changes")
    arms = dict(parse_arm(spec) for spec in args.arm)
    if len(arms) != len(args.arm):
        raise SystemExit("duplicate arm names")
    tasks_root = args.tasks_root.resolve()
    names = (
        [line.strip() for line in args.tasks_file.read_text().splitlines() if line.strip()]
        if args.tasks_file
        else sorted(p.name for p in tasks_root.iterdir() if (p / "task.toml").is_file())
    )
    gpu = {name for name in names if requires_gpu(tasks_root / name)}
    for name in names:
        require_guarded_verifier(tasks_root / name)
        if not sif_path(tasks_root / name).is_file():
            raise SystemExit(f"{name}: missing SIF {sif_path(tasks_root / name)}")
    slots = interleave(names, list(arms), args.draws, args.seed)
    for slot in slots:
        slot["pool"] = "gpu" if slot["task"] in gpu else "cpu"
    pools = {pool: [s for s in slots if s["pool"] == pool] for pool in ("cpu", "gpu")}
    runtime = block / "runtime"
    runtime.mkdir(parents=True)
    shared = args.dsh_runtime.resolve()
    (runtime / "node_modules").symlink_to(shared / "node_modules")
    for name in ("package.json", "package-lock.json"):
        (runtime / name).write_bytes((shared / name).read_bytes())
    bridge = HARNESS / "contree/sandbox-bridge.mjs"
    (runtime / "sandbox-bridge.mjs").write_bytes(bridge.read_bytes())
    dsh_version = json.loads((shared / "node_modules/@deepseek-ai/dsh/package.json").read_text())[
        "version"
    ]
    design = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code": identity,
        "tasks_root": str(tasks_root),
        "tasks": names,
        "gpu_tasks": sorted(gpu),
        "arms": arms,
        "draws": args.draws,
        "seed": args.seed,
        "budget": EPISODE_BUDGET,
        "env_file": str(args.env_file.resolve()),
        "dsh_version": dsh_version,
        "bridge_sha256": sha256_file(bridge),
        "profile_sha256": sha256_file(HARNESS / "contree/dsh.patch.yml.in"),
        "persona_sha256": sha256_file(HARNESS / "contree/persona-reference.txt"),
        "pools": {pool: len(items) for pool, items in pools.items()},
    }
    (block / "design.json").write_text(json.dumps(design, indent=2))
    for pool, items in pools.items():
        (block / f"slots-{pool}.json").write_text(json.dumps(items, indent=2))
    print(json.dumps({"block": str(block), **design["pools"], "arms": list(arms)}, indent=2))


def freeze(args: argparse.Namespace) -> None:
    from notes.dsh_inkling.harness.rivanna.apptainer import sif_path

    block: Path = args.block.resolve()
    design = json.loads((block / "design.json").read_text())
    out = block / "sif-sha256.json"
    if out.exists():
        raise SystemExit(f"refusing: {out} exists")
    root = Path(design["tasks_root"])
    digests = {}
    for name in design["tasks"]:
        path = sif_path(root / name)
        digests[name] = {"path": str(path), "sha256": sha256_file(path)}
    out.write_text(json.dumps(digests, indent=2))
    print(f"froze {len(digests)} SIF digests")


def run_slot(args: argparse.Namespace) -> None:
    block: Path = args.block.resolve()
    design = json.loads((block / "design.json").read_text())
    if git_identity(REPO_ROOT)["commit"] != design["code"]["commit"]:
        raise SystemExit("code checkout does not match the frozen design commit")
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    slot = json.loads((block / f"slots-{args.pool}.json").read_text())[index]
    arm = design["arms"][slot["arm"]]
    task = slot["task"]
    output = block / "runs" / slot["run"] / "results" / task
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}; episodes are never rerun silently")
    frozen = json.loads((block / "sif-sha256.json").read_text())[task]
    from notes.dsh_inkling.harness.rivanna.apptainer import sif_path

    tasks_root = Path(design["tasks_root"])
    if str(sif_path(tasks_root / task)) != frozen["path"] or (
        sha256_file(Path(frozen["path"])) != frozen["sha256"]
    ):
        raise SystemExit(f"{task}: SIF does not match the frozen digest")
    output.mkdir(parents=True)
    (output / "slot.json").write_text(
        json.dumps({**slot, "slurm_job": os.environ.get("SLURM_JOB_ID")}, indent=2)
    )
    budget = design["budget"]
    env_file = Path(design["env_file"])
    if arm["kind"] == "native":
        import asyncio

        from notes.dsh_inkling.harness.rivanna.run_native_episode import run

        asyncio.run(
            run(
                task=task,
                output=output,
                tasks_root=tasks_root,
                code_root=REPO_ROOT,
                env_file=env_file,
                grader_timeout=budget["grader_timeout"],
                sandbox_timeout=budget["sandbox_timeout"],
                max_infra_retries=budget["native_max_infra_retries"],
            )
        )
        return
    command = [
        sys.executable,
        str(HARNESS / "contree/run_task_contree.py"),
        task,
        str(output),
        "--sandbox",
        "apptainer",
        "--tasks-root",
        str(tasks_root),
        "--code-root",
        str(REPO_ROOT),
        "--runtime",
        str(block / "runtime"),
        "--proxy-script",
        str(HARNESS / "contree/proxy.py"),
        "--env-file",
        str(env_file),
        "--upstream",
        UPSTREAM,
        "--profile-patch",
        str(HARNESS / "contree/dsh.patch.yml.in"),
        "--persona-file",
        str(HARNESS / "contree/persona-reference.txt"),
        "--cache-path",
        str(block / "unused-contree-cache.json"),
        "--cpu-affinity",
        "none",
        "--max-turns",
        str(budget["max_model_requests"]),
        "--max-tool-calls",
        str(budget["max_tool_calls"]),
        "--wall-seconds",
        str(budget["wall_seconds"]),
        "--command-timeout",
        str(budget["command_timeout"]),
        "--grader-timeout",
        str(budget["grader_timeout"]),
        "--sandbox-timeout",
        str(budget["sandbox_timeout"]),
        *arm["flags"],
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if not (output / "summary.json").is_file():
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "task_name": task,
                    "harness_status": f"driver_exit_{completed.returncode}",
                    "reward": 0.0,
                    "error": "driver produced no summary",
                },
                indent=2,
            )
        )


def status(args: argparse.Namespace) -> None:
    block: Path = args.block.resolve()
    design = json.loads((block / "design.json").read_text())
    expected: Counter = Counter()
    for pool in ("cpu", "gpu"):
        for slot in json.loads((block / f"slots-{pool}.json").read_text()):
            expected[slot["arm"]] += 1
    done: dict[str, Counter] = defaultdict(Counter)
    for summary in block.glob("runs/*/results/*/summary.json"):
        arm = summary.parents[2].name.rsplit("-d", 1)[0]
        record = json.loads(summary.read_text())
        done[arm]["finished"] += 1
        done[arm]["errors"] += bool(record.get("error"))
        done[arm]["solved"] += record.get("reward") == 1.0
    started = Counter(
        p.parents[2].name.rsplit("-d", 1)[0] for p in block.glob("runs/*/results/*/slot.json")
    )
    rows = {}
    for arm in design["arms"]:
        finished = done[arm]["finished"]
        valid = finished - done[arm]["errors"]
        rows[arm] = {
            "expected": expected[arm],
            "started": started[arm],
            "finished": finished,
            "infra_errors": done[arm]["errors"],
            "solved": done[arm]["solved"],
            "solve_rate_valid": round(100 * done[arm]["solved"] / valid, 1) if valid else None,
        }
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("block", type=Path)
    p.add_argument("--tasks-root", type=Path, required=True)
    p.add_argument("--tasks-file", type=Path)
    p.add_argument("--arm", action="append", required=True, help="name=flags or name=native")
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--dsh-runtime", type=Path, required=True)
    p.add_argument("--allow-dirty", action="store_true")
    p.set_defaults(func=plan)
    p = sub.add_parser("freeze")
    p.add_argument("block", type=Path)
    p.set_defaults(func=freeze)
    p = sub.add_parser("run")
    p.add_argument("block", type=Path)
    p.add_argument("--pool", choices=("cpu", "gpu"), required=True)
    p.add_argument("--index", type=int)
    p.set_defaults(func=run_slot)
    p = sub.add_parser("status")
    p.add_argument("block", type=Path)
    p.set_defaults(func=status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
