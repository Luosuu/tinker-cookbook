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
    assert parse_arm("pi=pi") == ("pi", {"kind": "pi", "flags": [], "options": {}})
    assert parse_arm("pi_hook=pi hpc_mode=hook hpc_skill=swe-kokkos-debug") == (
        "pi_hook",
        {
            "kind": "pi",
            "flags": [],
            "options": {"hpc_mode": "hook", "hpc_skill": "swe-kokkos-debug"},
        },
    )
    with pytest.raises(ValueError):
        parse_arm("pi=pi hook")


def test_pi_agent_is_frozen_into_the_block(tmp_path: Path) -> None:
    """Updating the shared Pi checkout must not change a running block's agent."""
    from notes.dsh_inkling.harness.rivanna.block import freeze_pi_agent, pi_manifest

    source = tmp_path / "pi-src"
    (source / "node_modules").mkdir(parents=True)
    (source / "runner.mjs").write_text("v1")
    (source / "package.json").write_text('{"version": "0.3.0"}')
    (source / ".gitignore").write_text("node_modules/\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "v1"],
    ):
        subprocess.run(["git", "-C", str(source), *args], check=True)
    frozen = freeze_pi_agent(source, tmp_path / "block" / "pi")
    (source / "runner.mjs").write_text("v2")
    pi_dir = Path(frozen["dir"])
    assert (pi_dir / "runner.mjs").read_text() == "v1"
    assert (pi_dir / "node_modules").resolve() == (source / "node_modules").resolve()
    assert pi_manifest(pi_dir) == frozen["manifest"] and "runner.mjs" in frozen["manifest"]
    (pi_dir / "runner.mjs").write_text("tampered")
    assert pi_manifest(pi_dir) != frozen["manifest"]


def test_pi_arm_keeps_the_native_budget(tmp_path: Path) -> None:
    """A Pi arm may choose its integration, never its budget or inference settings."""
    from notes.dsh_inkling.harness.rivanna.run_pi_episode import pi_config

    config = pi_config(pi_agent_dir=tmp_path, grader_timeout=2400, sandbox_timeout=7200, options={})
    assert (config.max_turns, config.max_tool_calls, config.max_tool_output_chars) == (
        40,
        80,
        16384,
    )
    assert (config.thinking_effort, config.temperature, config.max_protocol_repairs) == (
        0.9,
        1.0,
        0,
    )
    with pytest.raises(ValueError, match="unsupported"):
        pi_config(
            pi_agent_dir=tmp_path, grader_timeout=1, sandbox_timeout=1, options={"max_turns": "80"}
        )


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
