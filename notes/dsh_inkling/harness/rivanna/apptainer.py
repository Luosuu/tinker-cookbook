"""Rivanna Apptainer backend shared by the DSH and native arms.

Both arms load the published v2.4 payload and open the same network-isolated task SIF with
the same thread settings and preflight as the audited v2.4 leaderboard run, so the harness
is the only difference between arms. The v2.4 verifier already embeds the runtime-test
guard and the command-level result collector; a payload without them must not be graded,
because the older published test.sh awards reward 1 when its filter selects zero tests.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

GUARD_MARKERS = ("Standalone runtime-test guard", "run_suite(json.loads(")

# Matches the v2.4 leaderboard worker: four threads on the four allocated CPUs.
COMMAND_ENV = {
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "CMAKE_BUILD_PARALLEL_LEVEL": "4",
    "OMP_PROC_BIND": "false",
}


def require_guarded_verifier(task_dir: Path) -> None:
    script = (task_dir / "tests" / "test.sh").read_text()
    missing = [marker for marker in GUARD_MARKERS if marker not in script]
    if missing:
        raise ValueError(
            f"{task_dir.name}: test.sh lacks the guarded v2.4 verifier ({', '.join(missing)}); "
            "refusing to grade with a verifier that can pass on zero selected tests"
        )


def load_guarded_task(tasks_root: Path, task: str):
    from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir

    by_name = {t.task_name: t for t in load_harbor_tasks_from_dir(tasks_root)}
    if task not in by_name:
        raise SystemExit(f"unknown task {task}")
    harbor_task = by_name[task]
    require_guarded_verifier(harbor_task.task_dir)
    return harbor_task


def sif_path(task_dir: Path) -> Path:
    return Path((task_dir / "environment" / "sif.path").read_text().strip())


def requires_gpu(task_dir: Path) -> bool:
    import tomllib

    config = tomllib.loads((task_dir / "task.toml").read_text())
    return int(config.get("environment", {}).get("gpus", 0)) > 0


def preflight_command(hidden_paths: tuple[str, ...]) -> str:
    paths = ("/tests", "/solution", *hidden_paths)
    return " && ".join(f"test ! -e {shlex.quote(p)}" for p in paths)


async def open_sandbox(harbor_task, timeout: int, *, hidden_paths: tuple[str, ...] = ()):
    """Open the task SIF offline and refuse it if hidden assets or reference history leak."""
    from tinker_cookbook.sandbox.apptainer_sandbox import apptainer_sandbox_factory

    sandbox = await apptainer_sandbox_factory(
        harbor_task.task_dir / "environment",
        timeout,
        allow_network=False,
        command_env=COMMAND_ENV,
    )
    try:
        answer = await sandbox.run_command(preflight_command(hidden_paths), timeout=30)
        if answer.exit_code != 0:
            raise RuntimeError("preflight failed: hidden assets are visible in the sandbox")
        metadata = json.loads((harbor_task.task_dir / "metadata.json").read_text())
        merge = metadata.get("merge_commit")
        if merge:
            answer = await sandbox.run_command(
                f"git cat-file -e {shlex.quote(merge)} 2>/dev/null", timeout=30
            )
            if answer.exit_code == 0:
                raise RuntimeError("preflight failed: reference history is visible")
    except BaseException:
        await sandbox.cleanup()
        raise
    return sandbox
