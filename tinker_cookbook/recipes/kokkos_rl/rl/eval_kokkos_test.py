import os
from dataclasses import dataclass

import pytest

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult, summarize_pass_at_k
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file, select_tasks


def test_load_env_file_preserves_existing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KOKKOS_TEST_NEW='loaded'\nKOKKOS_TEST_EXISTING=file\n")
    monkeypatch.setenv("KOKKOS_TEST_EXISTING", "process")
    monkeypatch.delenv("KOKKOS_TEST_NEW", raising=False)

    load_env_file(env_file)

    assert os.environ["KOKKOS_TEST_NEW"] == "loaded"
    assert os.environ["KOKKOS_TEST_EXISTING"] == "process"


@dataclass
class _Task:
    task_name: str


def test_select_tasks_is_deterministic_and_rejects_unknown_names() -> None:
    tasks = [_Task("a"), _Task("b"), _Task("c")]

    assert [task.task_name for task in select_tasks(tasks, "c,a")] == ["a", "c"]
    with pytest.raises(ValueError, match="unknown task names"):
        select_tasks(tasks, "missing")


def test_summarize_pass_at_k_uses_all_eight_samples() -> None:
    results = [
        TaskResult(
            task_name="a",
            sample_index=index,
            reward=float(index < 2),
            reward_details={},
            turns_used=1,
            time_seconds=1.0,
        )
        for index in range(8)
    ]

    summary = summarize_pass_at_k(results, [1, 4, 8])

    assert summary["pass_at_k"] == {
        "1": pytest.approx(0.25),
        "4": pytest.approx(1.0 - 15.0 / 70.0),
        "8": pytest.approx(1.0),
    }
