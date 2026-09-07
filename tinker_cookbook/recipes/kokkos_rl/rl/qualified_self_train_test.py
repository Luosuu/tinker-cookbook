import json
import random
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks_from_dir
from tinker_cookbook.recipes.kokkos_rl.rl import qualified_self_train as qualified
from tinker_cookbook.recipes.kokkos_rl.rl import self_train
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot_test import task_with_metadata


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def qualified_fixture(tmp_path):
    names = [f"task-{i:03d}" for i in range(99)] + ["kokkos__pykokkos-422"]
    tasks = []
    for name in names:
        task = task_with_metadata(tmp_path / "tasks", name)
        (task.task_dir / "instruction.md").write_text(task.instruction)
        (task.task_dir / "task.toml").write_text("")
        tasks.append(task)
    hashes = {t.task_name: qualified.current_task_digest(t) for t in tasks}
    bundle = tmp_path / "bundle"
    source = write(bundle / "review_source.json", {"reviewed": True})
    records = {}
    for task in tasks:
        p = write(
            bundle / "pairs" / f"{task.task_name}.json",
            {
                "task": task.task_name,
                "task_hash": hashes[task.task_name],
                **asdict(qualified.policy_for_task(task, "runtime_v12")),
                "nop": 0,
                "oracle": 1,
                "passed": True,
                "runtime_environment_policy_version": "dockerfile_env_v1",
            },
        )
        records[task.task_name] = {
            "qualified": True,
            "successful_evidence": [qualified.pinned_file(p)],
        }
    write(bundle / "manifest.json", {"task_hashes": hashes})
    write(
        bundle / "qualification.json",
        {
            "task_hashes": hashes,
            "qualified": 100,
            "ready_for_sampling": True,
            "resource_policy_version": "runtime_v12",
            "required_environment_policies": {"kokkos__pykokkos-422": "dockerfile_env_v1"},
            "tasks": records,
        },
    )
    write(
        bundle / "coverage_review.json",
        {"task_hashes": hashes, "status": "complete", "blockers": []},
    )
    write(
        bundle / "scope_allowlist.json",
        {
            "approvals": {
                name: {
                    "task": name,
                    "task_hash": digest,
                    "status": "accepted",
                    "sources": [qualified.pinned_file(source)],
                }
                for name, digest in hashes.items()
            }
        },
    )
    return tasks, bundle


@pytest.mark.parametrize("change", ["scope", "resource", "environment", "payload", "missing_pair"])
def test_qualified_inputs_reject_changed_evidence(tmp_path, change):
    tasks, bundle = qualified_fixture(tmp_path)
    resources = qualified.qualified_evidence(tasks, bundle)["resources"]
    assert isinstance(resources, dict) and len(resources) == 100
    if change == "scope":
        write(bundle / "review_source.json", {"reviewed": False})
    elif change == "payload":
        (tasks[0].task_dir / "tests/test.sh").write_text("different")
    else:
        qpath = bundle / "qualification.json"
        q = json.loads(qpath.read_text())
        name = "kokkos__pykokkos-422" if change == "environment" else tasks[0].task_name
        if change == "missing_pair":
            q["tasks"][name]["successful_evidence"] = []
        else:
            path = bundle / "pairs" / f"{name}.json"
            value = json.loads(path.read_text())
            value["cpu" if change == "resource" else "runtime_environment_policy_version"] = "wrong"
            write(path, value)
            q["tasks"][name]["successful_evidence"] = [qualified.pinned_file(path)]
        write(qpath, q)
    with pytest.raises(ValueError):
        qualified.qualified_evidence(tasks, bundle)


def test_review_receipt_cannot_accept_changed_artifacts(tmp_path):
    artifact = write(tmp_path / "trajectory.json", {"tokens": [1, 2]})
    assert not qualified.review_ready(tmp_path, "pilot", [artifact])
    write(
        tmp_path / "pilot_review.json",
        {
            "status": "accepted",
            "evidence": qualified.pinned_file(tmp_path / "pilot_review_required.json"),
        },
    )
    assert qualified.review_ready(tmp_path, "pilot", [artifact])
    write(artifact, {"tokens": [1, 3]})
    with pytest.raises(ValueError, match="artifacts changed"):
        qualified.review_ready(tmp_path, "pilot", [artifact])


@pytest.mark.asyncio
async def test_resource_factory_preserves_gpu_and_cpu_policies(tmp_path, monkeypatch):
    from tinker_cookbook.recipes.kokkos_rl.rl import resource_sandbox

    tasks = [
        task_with_metadata(tmp_path, name)
        for name in ("kokkos__kokkos-8989", "kokkos__kokkos-8891", "other")
    ]
    modal = AsyncMock()
    primary = AsyncMock()
    monkeypatch.setattr(resource_sandbox, "_create_modal_sandbox", modal)
    factory = qualified.qualified_factory(primary, tasks, "runtime_v12")
    for task in tasks:
        await factory(task.task_dir / "environment", 3600)
    assert modal.await_args_list[0].kwargs["gpu"] == "L4"
    assert modal.await_args_list[1].kwargs["build_parallelism"] == 4
    primary.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["complete", "interrupt", "prior_empty_error", "new_empty_error"]
)
async def test_pilot_counts_toward_baseline_and_interrupted_slots_never_resample(
    tmp_path, monkeypatch, scenario
):
    from tinker_cookbook.sandbox import contree_polling, contree_sandbox

    _, bundle = qualified_fixture(tmp_path)
    tasks = load_harbor_tasks_from_dir(tmp_path / "tasks")
    random.Random(7).shuffle(tasks)
    split = write(
        tmp_path / "original_split.json",
        {"heldout": [t.task_name for t in tasks[:20]], "train": [t.task_name for t in tasks[20:]]},
    )
    monkeypatch.setattr(self_train, "load_env_file", lambda _: None)
    monkeypatch.setattr(
        contree_sandbox,
        "ContreeDockerfileSandboxFactory",
        lambda *a, **k: SimpleNamespace(_client=SimpleNamespace(config=None)),
    )
    monkeypatch.setattr(contree_polling, "create_polling_client", lambda *a: None)
    monkeypatch.setattr(qualified, "qualified_factory", lambda primary, tasks, version: primary)
    generated = []

    async def evaluate(config, selected, sandbox_factory):
        directory = tmp_path / "run/baseline"
        path = directory / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        done = {(r["task_name"], r["sample_index"]) for r in rows}
        assert config.max_infra_retries == 0 and config.sandbox_resource_policy is not None
        for task in selected:
            for i in range(config.num_samples):
                if (task.task_name, i) in done:
                    continue
                generated.append((task.task_name, i))
                row = self_train.TaskResult(task.task_name, i, 0, {}, 40, 1)
                if scenario == "new_empty_error":
                    row.error = ""
                rows.append(asdict(row))
                folder = directory / "rollouts" / f"{task.task_name}__{i:02d}"
                for name in ("trajectory.json", "messages.json", "patch.diff", "verifier.json"):
                    write(folder / name, {"sampled_tokens": 10})
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        names = {t.task_name for t in selected}
        return [self_train.TaskResult(**r) for r in rows if r["task_name"] in names]

    monkeypatch.setattr(self_train, "run_eval", evaluate)
    config = self_train.Config(
        output_dir=str(tmp_path / "run"),
        tasks_dir=str(tmp_path / "tasks"),
        qualified_bundle_dir=str(bundle),
        split_manifest=str(split),
    )
    if scenario == "new_empty_error":
        with pytest.raises(RuntimeError, match="Unresolved infrastructure errors"):
            await self_train.main(config)
        assert len(generated) == 4
        assert not (tmp_path / "run/pilot_review_required.json").exists()
        return
    await self_train.main(config)
    assert len(generated) == 4
    assert (
        json.loads((tmp_path / "run/status.json").read_text())["stage"] == "awaiting_pilot_review"
    )
    if scenario == "prior_empty_error":
        path = tmp_path / "run/baseline/results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["error"] = ""
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError, match="Ambiguous or failed prior evaluation"):
            await self_train.main(config)
        assert len(generated) == 4
    elif scenario == "interrupt":
        path = tmp_path / "run/baseline/results.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with pytest.raises(ValueError, match="Interrupted evaluation"):
            await self_train.main(config)
        assert len(generated) == 4
    else:
        write(
            tmp_path / "run/pilot_review.json",
            {
                "status": "accepted",
                "evidence": qualified.pinned_file(tmp_path / "run/pilot_review_required.json"),
            },
        )
        await self_train.main(config)
        assert len(generated) == len(set(generated)) == 80
        assert (
            json.loads((tmp_path / "run/status.json").read_text())["stage"]
            == "awaiting_baseline_review"
        )
