import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from tinker_cookbook.recipes.harbor_rl import eval as harbor_eval
from tinker_cookbook.recipes.harbor_rl.eval_state import prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.utils.ml_log import dump_config


def make_task(root: Path, name: str) -> HarborTask:
    directory = root / name
    (directory / "environment").mkdir(parents=True)
    (directory / "tests").mkdir()
    (directory / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (directory / "tests" / "test.sh").write_text("echo test\n")
    return HarborTask(name, "fix this", directory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "other"),
        ("checkpoint_url", "new-checkpoint"),
        ("temperature", 0.7),
        ("max_tokens", 12),
        ("sandbox_backend", "other"),
        ("allow_network", False),
    ],
)
def test_resume_rejects_different_trial_config(tmp_path, field, value):
    task = make_task(tmp_path, "task")
    config = dump_config(harbor_eval.EvalConfig())
    out = tmp_path / "results"
    prepare_eval_state(out, config, [task], evaluator="tinker")
    with pytest.raises(ValueError, match="configuration differs"):
        prepare_eval_state(out, {**config, field: value}, [task], evaluator="tinker")


@pytest.mark.parametrize("change", ["instruction", "config", "tests", "environment"])
def test_resume_rejects_changed_task(tmp_path, change):
    task = make_task(tmp_path, "task")
    out = tmp_path / "results"
    prepare_eval_state(out, {}, [task], evaluator="test")
    if change == "instruction":
        task = replace(task, instruction="different")
    elif change == "config":
        task = replace(task, config={"verifier": {"timeout_sec": 7}})
    else:
        path = next((task.task_dir / change).iterdir())
        path.write_text("different")
    with pytest.raises(ValueError, match="task content changed"):
        prepare_eval_state(out, {}, [task], evaluator="test")


def test_resume_allows_selection_and_scheduling_changes(tmp_path):
    tasks = [make_task(tmp_path, name) for name in ("a", "b")]
    out = tmp_path / "results"
    prepare_eval_state(out, {"num_samples": 8, "max_concurrency": 2}, tasks, evaluator="test")
    prepare_eval_state(out, {"num_samples": 2, "max_concurrency": 4}, tasks[:1], evaluator="test")
    assert len(json.loads((out / "eval_identity.json").read_text())["tasks"]) == 2


def test_legacy_results_cannot_be_silently_relabelled(tmp_path):
    (tmp_path / "results.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Legacy"):
        prepare_eval_state(tmp_path, {}, [], evaluator="test")


@pytest.mark.asyncio
async def test_harbor_resume_filters_tasks_and_sample_indices(tmp_path, monkeypatch):
    tasks = [make_task(tmp_path, name) for name in ("a", "b")]
    out = tmp_path / "results"
    config = harbor_eval.EvalConfig(resume_dir=str(out), num_samples=2, pass_at_k="1,2")
    prepare_eval_state(out, dump_config(config), tasks, evaluator="tinker-harbor")
    stored = [
        harbor_eval.TaskResult(name, index, 1.0, {}, 1, 1.0)
        for name in ("a", "b")
        for index in range(4)
    ]
    (out / "results.jsonl").write_text(
        "".join(json.dumps(harbor_eval.asdict(r)) + "\n" for r in stored)
    )
    monkeypatch.setattr(harbor_eval.tinker, "ServiceClient", Mock())
    monkeypatch.setattr(harbor_eval.tokenizer_utils, "get_tokenizer", Mock())
    monkeypatch.setattr(harbor_eval.model_info, "get_recommended_renderer_name", Mock())
    monkeypatch.setattr(harbor_eval, "get_renderer", Mock())
    monkeypatch.setattr(harbor_eval, "TinkerTokenCompleter", Mock())
    results = await harbor_eval.run_eval(config, tasks[:1])
    assert [(r.task_name, r.sample_index) for r in results] == [("a", 0), ("a", 1)]
    summary = json.loads((out / "result.json").read_text())
    assert summary["num_tasks"] == 1
    assert summary["num_rollouts"] == 2
