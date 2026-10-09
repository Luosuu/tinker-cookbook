from __future__ import annotations

import json
from pathlib import Path

import pytest

from notes.dsh_inkling.compare_runs import load_rewards, pass_at_k


def test_load_rewards_reads_dsh_and_native_layouts(tmp_path: Path) -> None:
    dsh = tmp_path / "dsh" / "results" / "task-a"
    dsh.mkdir(parents=True)
    (dsh / "summary.json").write_text(json.dumps({"task_name": "task-a", "reward": 1.0}))
    native = tmp_path / "native" / "20261009_000000"
    native.mkdir(parents=True)
    (native / "results.jsonl").write_text(
        json.dumps({"task_name": "task-a", "reward": 0.0})
        + "\n"
        + json.dumps({"task_name": "task-b", "reward": 1.0})
        + "\n"
    )

    assert load_rewards(tmp_path / "dsh") == {"task-a": [1.0]}
    assert load_rewards(tmp_path / "native") == {"task-a": [0.0], "task-b": [1.0]}


def test_pass_at_k_is_the_unbiased_estimator() -> None:
    rewards = {"a": [1.0, 0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0, 0.0]}

    assert pass_at_k(rewards, 1) == pytest.approx(0.125)
    assert pass_at_k(rewards, 2) == pytest.approx(0.25)
    assert pass_at_k(rewards, 4) == pytest.approx(0.5)
    assert pass_at_k(rewards, 5) is None
