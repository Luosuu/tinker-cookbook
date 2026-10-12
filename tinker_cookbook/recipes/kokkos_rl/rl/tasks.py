"""Prepare current verification scripts without rewriting published datasets."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask, load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor import _dockerfile, _test_script
from tinker_cookbook.recipes.kokkos_rl.dataset.models import KokkosInstance
from tinker_cookbook.recipes.kokkos_rl.harness.profile import apply_profile, profile_text


@contextmanager
def prepared_kokkos_tasks(
    tasks_dir: Path, *, harness_profile: str = "baseline"
) -> Iterator[list[HarborTask]]:
    """Use the fixed verifier even for historical exports with mutable HEAD guards.

    Preserve instructions, hidden tests, and published evidence. For historical exports,
    regenerate only the environment and verifier from each task's metadata, in a temporary
    copy kept alive for the entire training/evaluation run; changed Dockerfile bytes also
    invalidate the prepared sandbox image cache. Payloads that already carry the guarded
    verifier (v2.4 and later) are used as published: regenerating them would drop their
    release repairs.
    """
    profile_text(harness_profile)  # Validate before preparing any task.
    with tempfile.TemporaryDirectory(prefix="kokkos-tasks-") as directory:
        tasks: list[HarborTask] = []
        for task in load_harbor_tasks_from_dir(tasks_dir):
            instance = KokkosInstance.from_dict(
                json.loads((task.task_dir / "metadata.json").read_text())
            )
            destination = Path(directory) / task.task_name
            shutil.copytree(task.task_dir, destination)
            if not is_guarded(task.task_dir):
                (destination / "environment" / "Dockerfile").write_text(_dockerfile(instance))
                (destination / "tests" / "test.sh").write_text(_test_script(instance))
            tasks.append(apply_profile(replace(task, task_dir=destination), harness_profile))
        yield tasks


GUARD_MARKERS = ("Standalone runtime-test guard", "run_suite(json.loads(")


def is_guarded(task_dir: Path) -> bool:
    script = (task_dir / "tests" / "test.sh").read_text()
    return all(marker in script for marker in GUARD_MARKERS)


def require_guarded_verifier(task_dir: Path) -> None:
    """Refuse a payload whose verifier could pass on zero selected tests.

    Payloads from v2.4 on embed the runtime-test guard and command-level collector and are
    graded as published; older published test.sh files must go through
    ``prepared_kokkos_tasks`` instead.
    """
    script = (task_dir / "tests" / "test.sh").read_text()
    missing = [marker for marker in GUARD_MARKERS if marker not in script]
    if missing:
        raise ValueError(
            f"{task_dir.name}: test.sh lacks the guarded verifier ({', '.join(missing)}); "
            "refusing to grade with a verifier that can pass on zero selected tests"
        )


def guarded_payload_tasks(tasks_dir: Path) -> list[HarborTask]:
    tasks = load_harbor_tasks_from_dir(tasks_dir)
    for task in tasks:
        require_guarded_verifier(task.task_dir)
    return tasks
