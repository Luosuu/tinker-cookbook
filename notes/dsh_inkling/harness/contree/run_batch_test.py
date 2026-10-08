from __future__ import annotations

import json
from pathlib import Path

import pytest

from notes.dsh_inkling.harness.contree.run_batch import _partition_for_resume


def test_resume_keeps_completed_and_cleans_interrupted_tasks(tmp_path: Path) -> None:
    results = tmp_path / "results"
    completed = results / "task-complete"
    interrupted = results / "task-interrupted"
    completed.mkdir(parents=True)
    interrupted.mkdir(parents=True)
    summary = {"task_name": "task-complete", "reward": 1.0}
    (completed / "summary.json").write_text(json.dumps(summary))
    (completed / "events.jsonl").write_text("keep")
    (interrupted / "events.jsonl").write_text("partial")

    pending, preserved = _partition_for_resume(
        ["task-complete", "task-interrupted", "task-not-started"], results
    )

    assert pending == [(1, "task-interrupted"), (2, "task-not-started")]
    assert preserved == {0: summary}
    assert (completed / "events.jsonl").read_text() == "keep"
    assert not interrupted.exists()


def test_resume_rejects_summary_for_the_wrong_task(tmp_path: Path) -> None:
    output = tmp_path / "results" / "expected-task"
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps({"task_name": "other-task"}))

    with pytest.raises(ValueError, match="invalid summary"):
        _partition_for_resume(["expected-task"], tmp_path / "results")
