from __future__ import annotations

import json
from pathlib import Path

import pytest

from notes.dsh_inkling.block_report import draw_solve_rates, welch


def _run(dir_: Path, rewards: list[tuple[str, float, str | None]]) -> Path:
    results = dir_ / "results"
    for task, reward, error in rewards:
        ep = results / task
        ep.mkdir(parents=True)
        (ep / "summary.json").write_text(
            json.dumps({"task_name": task, "reward": reward, "error": error})
        )
    return dir_


def test_draw_solve_rate_counts_all_episodes(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "r",
        [("a", 1.0, None), ("b", 0.0, None), ("c", 1.0, None), ("d", 0.0, None)],
    )
    assert draw_solve_rates(run, drop_infra=False) == 50.0


def test_draw_solve_rate_drops_infra_but_keeps_systemic_timeout(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "r",
        [
            ("a", 1.0, None),
            ("b", 0.0, "ApiTimeoutError: verifier unavailable"),
            ("kokkos__kokkos-8182", 0.0, "RuntimeError: verifier command failed"),
        ],
    )
    # as recorded: 1 of 3 solved
    assert draw_solve_rates(run, drop_infra=False) == pytest.approx(100 / 3)
    # dropping the generic infra error leaves a (solved) and 8182 (kept, unsolved): 1 of 2
    assert draw_solve_rates(run, drop_infra=True) == 50.0


def test_welch_detects_a_clean_separation() -> None:
    delta, t, _df, p = welch([45.0, 47.0, 49.0, 51.0], [52.0, 55.0, 50.0, 50.0])
    assert delta == pytest.approx(3.75)
    assert t > 0
    assert 0.0 <= p <= 1.0


def test_welch_is_degenerate_with_too_few_draws() -> None:
    delta, t, df, p = welch([50.0], [53.0])
    assert delta == pytest.approx(3.0)
    assert t != t  # nan
