import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tinker_cookbook.recipes.harbor_rl.eval_state import prepare_eval_state
from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
from tinker_cookbook.recipes.kokkos_rl.rl import eval_kokkos_openai as evaluation
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import (
    CLIConfig,
    _function_calls,
    _request_cost_usd,
    _truncate_tool_output,
    _usage,
)
from tinker_cookbook.utils.ml_log import dump_config


def test_function_calls_filters_non_tool_output() -> None:
    call = SimpleNamespace(type="function_call")
    response = SimpleNamespace(output=[SimpleNamespace(type="reasoning"), call])
    assert _function_calls(response) == [call]


def test_usage_extracts_reasoning_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(cached_tokens=5, cache_write_tokens=4),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )
    assert _usage(response) == (12, 7, 3, 5, 4)


def test_request_cost_distinguishes_cache_reads_and_writes() -> None:
    config = CLIConfig(
        input_price_per_million=2.0,
        cached_input_price_per_million=0.2,
        cache_write_price_per_million=2.5,
        output_price_per_million=12.0,
    )
    cost = _request_cost_usd(
        input_tokens=1_000_000,
        cached_input_tokens=600_000,
        cache_write_input_tokens=100_000,
        output_tokens=100_000,
        config=config,
    )
    assert cost == 2.17


def test_tool_output_truncation_preserves_head_and_tail() -> None:
    output = "a" * 100 + " important tail"
    truncated = _truncate_tool_output(output, 60)
    assert len(truncated) == 60
    assert truncated.startswith("a")
    assert truncated.endswith(" important tail")
    assert "characters omitted" in truncated


# Exercise orchestration with a fake API; each attempt still writes the real
# on-disk ledger, so retries and process restarts share the same accounting.


def _result(name: str, cost: float, error: str | None = None):
    return evaluation.OpenAITaskResult(
        task_name=name,
        reward=0.0 if error else 1.0,
        reward_details={},
        turns_used=1,
        tool_calls=1,
        input_tokens=100,
        output_tokens=10,
        reasoning_tokens=3,
        time_seconds=1.0,
        stop_reason="error" if error else "model_finished",
        estimated_cost_usd=cost,
        error=error,
    )


def _fake_attempts(monkeypatch, outcomes, prior_costs):
    monkeypatch.setattr("openai.AsyncOpenAI", Mock())

    async def evaluate(task, client, factory, config, results_dir, lock, prior_cost_usd=0.0):
        prior_costs.append(prior_cost_usd)
        result = outcomes.pop(0)
        with (results_dir / "results.jsonl").open("a") as file:
            file.write(json.dumps(asdict(result)) + "\n")
        return result

    monkeypatch.setattr(evaluation, "evaluate_task", evaluate)


@pytest.mark.asyncio
async def test_retry_and_resume_accumulate_all_attempt_costs(tmp_path, monkeypatch):
    task = make_task(tmp_path, "task")
    out = tmp_path / "results"
    config = CLIConfig(resume_dir=str(out), max_infra_retries=1, max_cost_usd_per_task=2.0)
    prior_costs = []
    _fake_attempts(
        monkeypatch, [_result("task", 0.7, "grader failed"), _result("task", 0.6)], prior_costs
    )
    await evaluation.run_eval(config, [task], Mock())
    summary = json.loads((out / "result.json").read_text())
    assert summary["estimated_cost_usd"] == pytest.approx(1.3)
    assert summary["input_tokens"] == 200
    assert summary["num_attempts"] == 2
    assert prior_costs == [0.0, 0.7]
    # A later invocation must retain the failed attempt's usage too.
    await evaluation.run_eval(config, [task], Mock())
    assert json.loads((out / "result.json").read_text()) == summary
    assert len(prior_costs) == 2


@pytest.mark.asyncio
async def test_exhausted_budget_does_not_reset_on_retry_or_resume(tmp_path, monkeypatch):
    task = make_task(tmp_path, "task")
    out = tmp_path / "results"
    config = CLIConfig(resume_dir=str(out), max_infra_retries=2, max_cost_usd_per_task=0.75)
    prior_costs = []
    _fake_attempts(monkeypatch, [_result("task", 0.8, "grader failed")], prior_costs)
    for _ in range(2):
        results = await evaluation.run_eval(config, [task], Mock())
        assert results[0].error == "grader failed"
    assert prior_costs == [0.0]
    summary = json.loads((out / "result.json").read_text())
    assert summary["estimated_cost_usd"] == 0.8
    assert summary["num_errors"] == 1
    assert summary["num_attempts"] == 1


@pytest.mark.asyncio
async def test_openai_resume_filters_old_tasks_and_checks_identity(tmp_path, monkeypatch):
    tasks = [make_task(tmp_path, name) for name in ("a", "b")]
    out = tmp_path / "results"
    config = CLIConfig(resume_dir=str(out))
    prepare_eval_state(out, dump_config(config), tasks, evaluator="openai-kokkos")
    (out / "results.jsonl").write_text(
        "".join(json.dumps(asdict(_result(t.task_name, 0.2))) + "\n" for t in tasks)
    )
    monkeypatch.setattr("openai.AsyncOpenAI", Mock())
    results = await evaluation.run_eval(config, tasks[:1], Mock())
    assert [r.task_name for r in results] == ["a"]
    assert json.loads((out / "result.json").read_text())["estimated_cost_usd"] == 0.2
    with pytest.raises(ValueError, match="configuration differs"):
        await evaluation.run_eval(CLIConfig(resume_dir=str(out), model_name="other"), tasks, Mock())
