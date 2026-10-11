from __future__ import annotations

import json
from pathlib import Path

import pytest

from notes.pi_harness.combine import combine, t_two_sided_p


@pytest.mark.parametrize(
    "t,df,p",
    [(2.0, 99, 0.04824), (2.5706, 5, 0.05), (1.0, 10, 0.34089), (0.0, 7, 1.0), (4.0, 3, 0.02799)],
)
def test_student_t_p_values(t: float, df: int, p: float) -> None:
    assert t_two_sided_p(t, df) == pytest.approx(p, abs=2e-4)


def _write(block: Path, run: str, rewards: dict[str, float], error_task: str | None = None) -> None:
    for task, reward in rewards.items():
        d = block / "runs" / run / "results" / task
        d.mkdir(parents=True)
        error = "502" if task == error_task else None
        (d / "summary.json").write_text(
            json.dumps({"task_name": task, "reward": reward, "error": error})
        )


def test_tasks_are_paired_within_blocks(tmp_path: Path) -> None:
    tasks = [f"t{i}" for i in range(20)]
    for b, shift in (("b1", 0), ("b2", 1)):
        block = tmp_path / b
        for d in range(4):
            # Control solves the first 10 tasks; treatment also solves t10 and t11.
            _write(
                block,
                f"native-d{d}",
                {t: float(i < 10) for i, t in enumerate(tasks)},
                error_task="t5" if d == 0 else None,
            )
            _write(
                block, f"pi-d{d}", {t: float(i < 12 - shift * (d % 2)) for i, t in enumerate(tasks)}
            )
    out = combine([tmp_path / "b1", tmp_path / "b2"], "pi", "native")
    paired = out["task_paired"]
    assert paired["tasks"] == 20
    # b1: +2 tasks in every draw; b2: +2 in even draws, +1 in odd draws → 1.5 tasks.
    assert paired["delta_points"] == pytest.approx(100 * (2 + 1.5) / 2 / 20)
    assert paired["bootstrap_95"][0] <= paired["delta_points"] <= paired["bootstrap_95"][1]
    assert [b["block"] for b in out["blocks"]] == ["b1", "b2"]
    assert out["blocks"][0]["control_draws"][0] == pytest.approx(100 * 9 / 19, abs=0.1)
