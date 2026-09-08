"""Regression coverage for verifier cancellation and retained retry evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.eval import (
    _evaluation_termination,
    _reserve_rollout_directory,
)
from tinker_cookbook.recipes.kokkos_rl.rl.rollout_data import _write_new_record
from tinker_cookbook.rl.rollout_presets import agentic


def test_retry_keeps_first_sampling_evidence(tmp_path: Path) -> None:
    root = tmp_path / "rollouts" / "task__00"
    first = _reserve_rollout_directory(root)
    _write_new_record(first / "sampling/000.request.json", {"attempt": 0})
    second = _reserve_rollout_directory(root)
    _write_new_record(second / "sampling/000.request.json", {"attempt": 1})
    assert first == root
    assert second == root / "attempts/001"
    assert '"attempt": 0' in (first / "sampling/000.request.json").read_text()


def test_attempt_reservation_is_atomic(tmp_path: Path) -> None:
    root = tmp_path / "task__00"
    with ThreadPoolExecutor(max_workers=8) as pool:
        directories = list(pool.map(lambda _: _reserve_rollout_directory(root), range(16)))
    assert len(set(directories)) == 16
    assert all(p.is_dir() for p in directories)


def test_harbor_verifier_owns_timeout() -> None:
    preset = agentic().termination
    policy = _evaluation_termination()
    assert preset.grader_timeout_seconds == 900
    assert policy.grader_timeout_seconds is None
    assert policy.zero_reward_on_limit == preset.zero_reward_on_limit
    assert policy.skip_grading_on_timeout == preset.skip_grading_on_timeout
