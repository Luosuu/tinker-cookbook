"""Regression coverage for verifier cancellation and retained retry evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.eval import (
    _evaluation_termination,
    _publish_completed_attempt,
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


def test_successful_retry_remains_visible_to_legacy_readers(tmp_path: Path) -> None:
    from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
    from tinker_cookbook.recipes.kokkos_rl.rl.self_train import delivery_metrics

    root = tmp_path / "rollouts/task__00"
    first = _reserve_rollout_directory(root)
    _write_new_record(first / "sampling/000.request.json", {"original": True})
    _write_new_record(first / "failure.json", {"error": "timeout"})
    retry = _reserve_rollout_directory(root)
    _write_new_record(retry / "sampling/000.request.json", {"retry": True})
    _write_new_record(retry / "trajectory.json", {"sampled_tokens": 42})
    _publish_completed_attempt(root, retry)
    result = TaskResult("task", 0, 1.0, {}, 2, 1.0)
    assert delivery_metrics([result], tmp_path)["sampled_tokens_recorded"] == 42
    assert list((tmp_path / "rollouts").glob("*/trajectory.json")) == [root / "trajectory.json"]
    assert (root / "attempts/000/failure.json").exists()
    assert (root / "attempts/000/sampling/000.request.json").exists()
    assert not (root / "failure.json").exists()
    assert (root / "sampling").resolve() == retry / "sampling"
