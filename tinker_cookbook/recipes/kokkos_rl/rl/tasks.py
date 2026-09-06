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


@contextmanager
def prepared_kokkos_tasks(tasks_dir: Path) -> Iterator[list[HarborTask]]:
    """Use the fixed verifier even for historical exports with mutable HEAD guards.

    Preserve instructions, hidden tests, and published evidence. Regenerate only
    the environment and verifier from each task's metadata, in a temporary copy
    kept alive for the entire training/evaluation run. Changed Dockerfile bytes
    also invalidate the prepared sandbox image cache.
    """
    with tempfile.TemporaryDirectory(prefix="kokkos-tasks-") as directory:
        tasks: list[HarborTask] = []
        for task in load_harbor_tasks_from_dir(tasks_dir):
            instance = KokkosInstance.from_dict(
                json.loads((task.task_dir / "metadata.json").read_text())
            )
            destination = Path(directory) / task.task_name
            shutil.copytree(task.task_dir, destination)
            (destination / "environment" / "Dockerfile").write_text(_dockerfile(instance))
            (destination / "tests" / "test.sh").write_text(_test_script(instance))
            tasks.append(replace(task, task_dir=destination))
        yield tasks
