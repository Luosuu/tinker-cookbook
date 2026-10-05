"""Tests for the DSH run analyzer.

The analyzer decides whether a run is comparable (prompt parity) and where its budget
went. Both feed the go/no-go decision for a harness variant, so the parsing is pinned
against fixture data rather than the live cluster.
"""

from __future__ import annotations

import json
from pathlib import Path

from notes.dsh_inkling.analyze_run import Episode, load_run


def _write_episode(
    run: Path,
    task: str,
    *,
    reward: float,
    status: str,
    commands: list[str],
    roles: list[str] | None = None,
    tools: list[str] | None = None,
    input_tokens: int = 1000,
) -> None:
    directory = run / "results" / task
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "reward": reward,
                "harness_status": status,
                "turns_used": len(commands),
                "time_seconds": 1.0,
                "error": None,
            }
        )
    )
    with (directory / "events.jsonl").open("w") as handle:
        for index, command in enumerate(commands, start=1):
            handle.write(json.dumps({"type": "bash", "index": index, "command": command}) + "\n")
    messages = [
        {"role": role, "chars": 164 if role == "system" else 800}
        for role in roles or ["system", "user"]
    ]
    with (directory / "proxy-budget.jsonl").open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "request",
                    "budget": {"requests": len(commands), "input_tokens": input_tokens},
                    "tools": tools or ["bash"],
                    "messages": messages,
                }
            )
            + "\n"
        )


def test_loads_episodes_with_commands_and_budget(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_episode(
        run,
        "t-1",
        reward=1.0,
        status="finished",
        commands=["grep -rn x .", "sed -i s/a/b/ f.hpp"],
        input_tokens=250_000,
    )
    episodes = load_run(run)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.task == "t-1"
    assert episode.reward == 1.0
    assert episode.input_tokens == 250_000
    assert episode.tools == ["bash"]
    assert episode.roles == ["system", "user"]


def test_detects_first_write_and_ignores_devnull_reads(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_episode(
        run,
        "t-1",
        reward=0.0,
        status="exit_1",
        commands=[
            "grep -rn x . 2>/dev/null",
            "ls -la /workspace/repo",
            "cat > /workspace/repo/f.hpp << 'EOF'",
        ],
    )
    episode = load_run(run)[0]
    assert episode.turn_of_first_write() == 2
    assert episode.wrote


def test_episode_with_no_write_is_reported_as_such(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_episode(
        run, "t-1", reward=0.0, status="exit_1", commands=["ls", "grep -rn y .", "find . -name x"]
    )
    episode = load_run(run)[0]
    assert episode.turn_of_first_write() is None
    assert not episode.wrote


def test_handles_missing_evidence_files(tmp_path: Path) -> None:
    """A crashed episode still has a summary; the analyzer must not raise on it."""
    run = tmp_path / "run"
    directory = run / "results" / "t-1"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps({"task_name": "t-1", "reward": 0.0, "harness_status": "not_started"})
    )
    episode = load_run(run)[0]
    assert episode.commands == []
    assert not episode.wrote


def test_load_run_ignores_non_directories(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    (run / "results" / "stray.txt").write_text("not an episode")
    assert load_run(run) == []


def test_episode_wrote_is_false_when_commands_empty() -> None:
    assert not Episode(task="t", reward=0.0, status="x", error=None, turns=0, seconds=0.0).wrote
