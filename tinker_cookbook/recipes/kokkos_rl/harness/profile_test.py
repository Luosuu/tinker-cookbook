from pathlib import Path

import pytest

from tinker_cookbook.recipes.harbor_rl.eval_state import prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.recipes.kokkos_rl.harness.profile import apply_profile


def test_profile_preserves_task_and_invalidates_resume(tmp_path: Path):
    task = HarborTask("task", "Fix a reduction.", tmp_path, {})
    assert apply_profile(task, "baseline") is task
    augmented = apply_profile(task, "hpc-debug-v1")
    assert augmented.instruction.endswith(task.instruction)
    assert task.config == {}
    prepare_eval_state(tmp_path, {}, [task], evaluator="test")
    with pytest.raises(ValueError):
        prepare_eval_state(tmp_path, {}, [augmented], evaluator="test")
    with pytest.raises(ValueError, match="Unknown harness profile"):
        apply_profile(task, "typo")
