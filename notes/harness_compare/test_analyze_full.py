"""Check pass@4 denominators and invalid-attempt accounting."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_full import analyze


def make_run(root: Path) -> list[str]:
    tasks = [f"task-{i:03d}" for i in range(100)]
    (root / "design.json").write_text(json.dumps({
        "tasks": tasks,
        "samples": 4,
        "splits": {"train": tasks[:80], "heldout": tasks[80:]},
        "image": {name: {"gpu": False} for name in tasks},
    }))
    for task in tasks:
        for sample in range(4):
            folder = root / "results" / task / f"sample-{sample}"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps({
                "task_name": task, "sample_index": sample,
                "reward": int(task == tasks[0] and sample == 2),
                "error": None, "time_seconds": 10, "turns_used": 3,
            }))
            (folder / "status.json").write_text(json.dumps({"model_turns_observed": 4}))
            for filename in ("candidate.patch", "grading.json", "preflight.json"):
                (folder / filename).write_text("{}")
    return tasks


def test_complete_and_incomplete_pass_at_4() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        tasks = make_run(root)
        complete = analyze(root)
        assert complete["pass_at_4"] == 0.01
        assert complete["pass_at_1"] == 1 / 400
        assert complete["complete_tasks"] == 100

        (root / "results" / tasks[1] / "sample-3" / "summary.json").unlink()
        (root / "results" / tasks[2] / "sample-1" / "grading.json").unlink()
        incomplete = analyze(root)
        assert incomplete["pass_at_4"] is None
        assert incomplete["pass_at_1"] is None
        assert incomplete["complete_tasks"] == 98
        assert incomplete["pass_at_4_lower_bound"] == 0.01
        assert incomplete["pass_at_4_upper_bound"] == 0.03
        assert incomplete["states"]["missing"] == 1
        assert incomplete["states"]["infrastructure_failure"] == 1
