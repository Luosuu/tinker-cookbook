import json

import httpx
import pytest

from tinker_cookbook.recipes.kokkos_rl.chat_inference import ChatSession
from tinker_cookbook.recipes.kokkos_rl.chat_inference_test import api_client, completion
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import (
    SweepConfig,
    model_config,
    summarize,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import BASH_TOOL, OpenAITaskResult


@pytest.mark.asyncio
async def test_nebius_reasoning_roundtrip_omits_tinker_fields_and_temperature():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=completion(calls=len(requests) == 1))

    async with api_client(handler) as client:
        session = ChatSession(client, "test", 0.9, None, provider="nebius", reasoning_effort="high")
        await session.create(
            [{"role": "user", "content": "hi"}],
            instructions="system",
            tools=[BASH_TOOL],
            max_tokens=100,
        )
        await session.create(
            [{"type": "function_call_output", "call_id": "call_1", "output": "/repo"}],
            instructions="system",
            tools=[BASH_TOOL],
            max_tokens=80,
        )
    assert requests[0]["reasoning_effort"] == "high"
    assert "separate_reasoning" not in requests[0]
    assert "temperature" not in requests[0]
    assert requests[1]["messages"][2]["reasoning_content"] == "retained reasoning"
    assert requests[1]["max_tokens"] == 80


def test_nebius_verified_prices_and_hard_budgets():
    config = model_config(
        "test", {"pricing": {"prompt": "0.000003", "completion": "0.000015"}}, SweepConfig()
    )
    assert config.max_turns == 40
    assert config.max_sampled_tokens == 65536
    assert config.input_price_per_million == 3
    assert config.cached_input_price_per_million == 3
    assert config.output_price_per_million == 15
    assert config.max_cost_usd_per_task is None
    assert config.max_infra_retries == 0


def test_incomplete_or_infra_errors_never_report_full_pass_at_one():
    result = OpenAITaskResult("task", 1, {}, 4, 3, 20, 5, None, 1, "model_finished")
    assert summarize([result], 100)["pass_at_1"] is None
    assert summarize([result], 1)["pass_at_1"] == 1
    result.error = "sandbox lost"
    summary = summarize([result], 1)
    assert summary["pass_at_1"] is None
    assert summary["errors"] == 1


@pytest.mark.asyncio
async def test_sweep_shared_limit_smoke_gate_and_no_resampling(tmp_path, monkeypatch):
    import asyncio
    from dataclasses import asdict

    from tinker_cookbook.recipes.harbor_rl.eval_state_test import make_task
    from tinker_cookbook.recipes.kokkos_rl.rl import eval_kokkos_nebius as sweep

    tasks = [make_task(tmp_path, name) for name in ("smoke", "next")]
    monkeypatch.setattr(sweep, "load_harbor_tasks_from_dir", lambda path: tasks)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"task_hashes": {t.task_name: sweep._task_digest(t) for t in tasks}})
    )
    monkeypatch.setenv("TEST_NEBIUS_KEY", "test")
    root, gate = tmp_path / "output", tmp_path / "gates"
    root.mkdir()
    gate.mkdir()
    for task in tasks:
        (gate / f"{task.task_name}.json").write_text('{"passed":true}')
    (root / "models.json").write_text(
        json.dumps(
            {
                "data": [
                    {"id": m, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
                    for m in sweep.MODELS
                ]
            }
        )
    )
    active = peak = 0
    calls = []
    finished = set()

    async def fake_evaluate(task, client, factory, config, directory, lock):
        nonlocal active, peak
        if task.task_name == "next":
            assert all((m, "smoke") in finished for m in sweep.MODELS)
        active += 1
        peak = max(peak, active)
        calls.append((config.model_name, task.task_name))
        await asyncio.sleep(0)
        result = OpenAITaskResult(task.task_name, 1, {}, 2, 1, 20, 5, None, 1, "model_finished")
        (directory / "results.jsonl").write_text(json.dumps(asdict(result)) + "\n")
        active -= 1
        finished.add((config.model_name, task.task_name))
        return result

    monkeypatch.setattr(sweep, "evaluate_task", fake_evaluate)
    config = SweepConfig(
        output_path=str(root),
        validation_dir=str(gate),
        api_key_env="TEST_NEBIUS_KEY",
        source_manifest=str(manifest_path),
        expected_tasks=2,
        smoke_task="smoke",
    )
    await sweep.main(config)
    await sweep.main(config)
    assert peak == 4
    assert len(calls) == 8
    assert len(set(calls)) == 8
    assert json.loads((root / "status.json").read_text())["state"] == "complete"
    (root / sweep.MODELS[0].split("/")[-1] / "next/results.jsonl").unlink()
    await sweep.main(config)
    assert len(calls) == 8
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "interrupted_needs_review"
    assert status["interrupted"] == [[sweep.MODELS[0], "next"]]
