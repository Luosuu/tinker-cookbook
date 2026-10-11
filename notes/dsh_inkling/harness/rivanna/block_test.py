"""Guards for the Rivanna block runner and its Apptainer backend."""

from __future__ import annotations

import shlex
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from notes.dsh_inkling.harness.rivanna.apptainer import (
    GUARD_MARKERS,
    load_guarded_task,
    preflight_command,
    require_guarded_verifier,
    requires_gpu,
)
from notes.dsh_inkling.harness.rivanna.block import interleave, parse_arm

TASKS = [f"kokkos__kokkos-{n}" for n in range(1000, 1010)]


def _task(root: Path, name: str, script: str, gpus: int = 0) -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "tests/test.sh").write_text(script)
    (task / "instruction.md").write_text("fix it")
    toml = '[environment]\nworkdir = "/workspace/repo"\n'
    if gpus:
        toml += f"gpus = {gpus}\n"
    (task / "task.toml").write_text(toml)
    return task


def test_arms_stay_level_throughout_the_queue() -> None:
    """At any point every arm has started the same number of episodes, plus or minus one."""
    arms = ["native", "control", "state"]
    slots = interleave(TASKS, arms, draws=4, seed=3)
    assert len(slots) == len(TASKS) * len(arms) * 4
    started: Counter = Counter()
    for slot in slots:
        started[slot["arm"]] += 1
        assert max(started.values()) - min(started[a] for a in arms) <= 1
    per_run = Counter((s["run"], s["task"]) for s in slots)
    assert set(per_run.values()) == {1}, "each run visits each task exactly once"


def test_interleave_is_reproducible_and_draws_differ() -> None:
    first = interleave(TASKS, ["a", "b"], draws=2, seed=7)
    assert first == interleave(TASKS, ["a", "b"], draws=2, seed=7)
    order = [[s["task"] for s in first if s["draw"] == d and s["arm"] == "a"] for d in (0, 1)]
    assert order[0] != order[1]


def test_arm_specs() -> None:
    assert parse_arm("native=native") == ("native", {"kind": "native", "flags": []})
    assert parse_arm("control=") == ("control", {"kind": "dsh", "flags": []})
    assert parse_arm("state_eb=--state-target --env-brief") == (
        "state_eb",
        {"kind": "dsh", "flags": ["--state-target", "--env-brief"]},
    )
    with pytest.raises(ValueError):
        parse_arm("bad name=--state-target")


def test_unguarded_payload_is_refused(tmp_path: Path) -> None:
    """The published v2 test.sh passes on zero selected tests and must never be graded."""
    _task(tmp_path, "kokkos__kokkos-1", "cmake --build build --target X\n")
    with pytest.raises(ValueError, match="guarded"):
        load_guarded_task(tmp_path, "kokkos__kokkos-1")
    _task(tmp_path, "kokkos__kokkos-2", "\n".join(GUARD_MARKERS))
    assert load_guarded_task(tmp_path, "kokkos__kokkos-2").task_name == "kokkos__kokkos-2"
    with pytest.raises(ValueError):
        require_guarded_verifier(_task(tmp_path, "kokkos__kokkos-3", GUARD_MARKERS[0]))


def test_gpu_tasks_are_detected_from_task_toml(tmp_path: Path) -> None:
    assert requires_gpu(_task(tmp_path, "cuda", "", gpus=1))
    assert not requires_gpu(_task(tmp_path, "cpu", ""))


def test_preflight_fails_when_any_hidden_path_exists(tmp_path: Path) -> None:
    secret = tmp_path / "inkling.env"
    command = preflight_command((str(secret),))
    assert (
        subprocess.run(
            [
                "bash",
                "-c",
                command.replace("/tests", "/nonexistent-a").replace("/solution", "/nonexistent-b"),
            ]
        ).returncode
        == 0
    )
    secret.write_text("TINKER_API_KEY=x")
    assert (
        subprocess.run(
            [
                "bash",
                "-c",
                command.replace("/tests", "/nonexistent-a").replace("/solution", "/nonexistent-b"),
            ]
        ).returncode
        != 0
    )
    assert shlex.quote(str(secret)) in command


def test_slurm_never_pins_cpus_inside_the_allocation() -> None:
    """taskset -c 0,1 inside a Slurm cgroup can name CPUs the job does not own."""
    source = (Path(__file__).parent / "block.py").read_text()
    assert '"--cpu-affinity",\n        "none",' in source
