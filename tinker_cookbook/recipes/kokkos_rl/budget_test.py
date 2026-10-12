import pytest

from tinker_cookbook.recipes.kokkos_rl.budget import BUDGETS, CONTEXT_WINDOW, LONG_250, V24, budget


def test_v24_is_the_frozen_benchmark_budget():
    assert (V24.max_turns, V24.max_tool_calls, V24.max_sampled_tokens, V24.max_input_tokens) == (
        40,
        80,
        65_536,
        5_000_000,
    )
    assert V24.max_context_tokens is None
    assert (V24.wall_seconds, V24.grader_seconds, V24.sandbox_seconds) == (3_600, 2_400, 7_200)


@pytest.mark.parametrize("profile", list(BUDGETS.values()), ids=list(BUDGETS))
def test_profiles_are_internally_consistent(profile):
    assert profile.max_tool_calls >= 2 * profile.max_turns
    assert profile.sandbox_seconds >= profile.wall_seconds + profile.grader_seconds
    assert profile.max_sampled_tokens >= profile.max_tokens
    assert profile.max_tool_output_chars == 16_384


def test_long_budget_binds_only_on_requests_and_context():
    assert LONG_250.max_turns == 250
    assert LONG_250.max_context_tokens == CONTEXT_WINDOW
    # Each prompt is bounded by the window, so the cumulative cap cannot bind first.
    assert LONG_250.max_input_tokens >= LONG_250.max_turns * CONTEXT_WINDOW


def test_unknown_profile_is_refused():
    assert budget("long-250") is LONG_250
    with pytest.raises(ValueError, match="unknown budget"):
        budget("long-200")


def test_long_budget_stays_above_measured_use():
    """block-t01 maxima: 102 requests, 25,191 output tokens, 1,787 s, 8 build checks (capped)."""
    assert LONG_250.max_turns >= 2 * 102
    assert LONG_250.max_sampled_tokens >= 4 * 25_191
    assert LONG_250.wall_seconds >= 3 * 1_787
    assert LONG_250.build_check_max_runs > 8
    assert LONG_250.build_check_max_seconds > 1_200


def test_v25_payload_is_graded_as_published(tmp_path):
    """Regenerating a guarded v2.4+ verifier would drop its release repairs."""
    import shutil
    from pathlib import Path

    from tinker_cookbook.recipes.kokkos_rl.rl.tasks import is_guarded, prepared_kokkos_tasks

    source = Path(__file__).parents[3] / "data/kokkos/SWE-kokkos-bench-v2.5/kokkos__kokkos-6289"
    shutil.copytree(source, tmp_path / source.name)
    assert is_guarded(source)
    with prepared_kokkos_tasks(tmp_path) as tasks:
        (task,) = tasks
        for rel in ("tests/test.sh", "environment/Dockerfile", "task.toml"):
            assert (task.task_dir / rel).read_bytes() == (source / rel).read_bytes(), rel
    import tomllib

    config = tomllib.loads((source / "task.toml").read_text())
    assert config["metadata"]["budget"] == LONG_250.name
    assert config["agent"]["timeout_sec"] == LONG_250.wall_seconds
    assert config["verifier"]["timeout_sec"] == LONG_250.grader_seconds
