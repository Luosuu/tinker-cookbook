import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl import donor_sft as ds


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def trajectory(task, index=0):
    return {
        "task_name": task,
        "sample_index": index,
        "reward": 1,
        "turns": index + 1,
        "sampled_tokens": 1,
        "stop_reason": "completed",
        "audit_flags": [],
        "datums": [{"input_tokens": [10, 11], "target_tokens": [11, 12], "weights": [0.0, 1.0]}],
    }


def dataset_fixture(tmp_path):
    paths = [write(tmp_path / f"task{i}.json", trajectory(f"task{i}")) for i in range(31)]
    content = {
        "paths": [str(p) for p in paths],
        "proofs": {str(p): ds.pinned_file(p) for p in paths},
        "train_tasks": [f"task{i}" for i in range(80)],
        "heldout_tasks": [f"heldout{i}" for i in range(20)],
    }
    manifest = write(tmp_path / "manifest.json", content)
    return manifest, content, paths


def test_raw_dataset_weights_and_budget(tmp_path):
    m, c, p = dataset_fixture(tmp_path)
    dataset, _ = ds.PinnedDatasetBuilder(
        manifest=str(m), manifest_sha256=ds.pinned_file(m)["sha256"]
    )()
    assert len(dataset) == 8
    assert (
        sum(
            sum(sum(d.loss_fn_inputs["weights"].data) for d in dataset.get_batch(i))
            for i in range(8)
        )
        == 31
    )
    datum = dataset.get_batch(0)[0]
    assert datum.model_input.to_ints() == [10, 11]
    assert datum.loss_fn_inputs["target_tokens"].data == [11, 12]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 1.0]
    with pytest.raises(ValueError):
        dataset.get_batch(8)
    p[0].write_text("{}")
    with pytest.raises(ValueError):
        dataset.get_batch(0)


@pytest.mark.parametrize(
    "change", ["heldout", "duplicate", "float_tokens", "manifest_sha", "changed_frozen"]
)
def test_builder_rejects_bad_input(tmp_path, change):
    m, c, p = dataset_fixture(tmp_path)
    if change == "heldout":
        c["heldout_tasks"].append("task0")
    elif change == "duplicate":
        c["paths"][1] = c["paths"][0]
    elif change == "float_tokens":
        row = json.loads(p[0].read_text())
        row["datums"][0]["input_tokens"][0] = 10.0
        write(p[0], row)
        c["proofs"][str(p[0])] = ds.pinned_file(p[0])
    elif change == "changed_frozen":
        p[0].write_text("{}")
    write(m, c)
    sha = "wrong" if change == "manifest_sha" else ds.pinned_file(m)["sha256"]
    with pytest.raises(ValueError):
        ds.PinnedDatasetBuilder(manifest=str(m), manifest_sha256=sha)()


def selection_fixture(tmp_path):
    rows = []
    for i in range(31):
        task = "kokkos__kokkos-7148" if i == 0 else f"task{i:02}"
        for j in range(3 if i < 28 else 2):
            # The excluded real slot must be extra, not one of the90 accepted.
            sample = 3 if i == 0 and j == 2 else j
            row = trajectory(task, sample)
            p = write(tmp_path / f"{task}__{sample:02d}.json", row)
            rows.append((p, row))
    excluded = trajectory("kokkos__kokkos-7148", 2)
    rows.append((write(tmp_path / "excluded.json", excluded), excluded))
    arms = ds.choose_donors([(p, r) for p, r in rows if r != excluded])
    summaries = {}
    for arm, paths in arms.items():
        chosen = [json.loads(Path(p).read_text()) for p in paths]
        summaries[arm] = {
            "slots": [f"{r['task_name']}__{r['sample_index']:02d}" for r in chosen],
            "tasks": 31,
            "turns": sum(r["turns"] for r in chosen),
            "sampled_tokens": 31,
            "datums": 31,
            "updates": 8,
        }
    review = {
        "status": "accepted_donor_selection_only",
        "accepted_candidates": 90,
        "excluded_slots": {ds.EXCLUDED: "503"},
        "arms": summaries,
    }
    return rows, review


def test_selection_same_support_and_excluded_slot(tmp_path):
    rows, review = selection_fixture(tmp_path)
    arms = ds.selection(rows, review)
    assert all(len(paths) == 31 for paths in arms.values())
    assert [json.loads(Path(p).read_text())["task_name"] for p in arms["random_success"]] == [
        json.loads(Path(p).read_text())["task_name"] for p in arms["short_success"]
    ]
    assert all(str(tmp_path / "excluded.json") not in paths for paths in arms.values())


@pytest.mark.parametrize("change", ["slot", "budget", "exclusion", "approval", "count"])
def test_selection_differs_from_receipt_refused(tmp_path, change):
    rows, review = selection_fixture(tmp_path)
    if change == "slot":
        review["arms"]["random_success"]["slots"][0] = "wrong"
    elif change == "budget":
        review["arms"]["short_success"]["updates"] = 9
    elif change == "exclusion":
        review["excluded_slots"]["extra"] = "bad"
    elif change == "approval":
        review["status"] = "pending"
    else:
        rows.pop()
    with pytest.raises(ValueError):
        ds.selection(rows, review)


def complete_training(log):
    log.mkdir(parents=True, exist_ok=True)
    (log / "checkpoints.jsonl").write_text(
        json.dumps(
            {
                "name": "final",
                "epoch": 1,
                "batch": 0,
                "state_path": "tinker://test/state",
                "sampler_path": "tinker://test/sampler",
            }
        )
        + "\n"
    )
    (log / "metrics.jsonl").write_text(
        "".join(
            json.dumps(
                {"step": i, "epoch": 0, "learning_rate": 1e-5, "num_loss_tokens": 4 if i < 7 else 3}
            )
            + "\n"
            for i in range(8)
        )
    )


@pytest.mark.parametrize("change", ["steps", "learning_rate", "weighting", "checkpoint", "sampler"])
def test_bad_training_evidence_refused(tmp_path, change):
    complete_training(tmp_path)
    if change in ("steps", "learning_rate", "weighting"):
        p = tmp_path / "metrics.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        if change == "steps":
            rows.pop()
        elif change == "learning_rate":
            rows[0]["learning_rate"] = 1e-4
        else:
            rows[0]["num_loss_tokens"] = 9
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        p = tmp_path / "checkpoints.jsonl"
        r = json.loads(p.read_text())
        r["epoch" if change == "checkpoint" else "sampler_path"] = (
            2 if change == "checkpoint" else None
        )
        write(p, r)
    with pytest.raises(ValueError):
        ds.training_evidence(tmp_path)


def test_train_once_and_complete_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "verify_bundle", lambda _: {})
    monkeypatch.setattr(ds, "training_config", lambda b, a, log: SimpleNamespace(log_path=str(log)))

    async def train(cfg):
        complete_training(Path(cfg.log_path))

    mock = AsyncMock(side_effect=train)
    monkeypatch.setattr(ds.train, "main", mock)

    async def go():
        a = await ds.train_arm(tmp_path / "bundle", "random_success", tmp_path / "arm", "id")
        b = await ds.train_arm(tmp_path / "bundle", "random_success", tmp_path / "arm", "id")
        assert a == b and a["stage"] == "complete"

    asyncio.run(go())
    assert mock.await_count == 1


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_failed_training_never_retries(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(ds, "verify_bundle", lambda _: {})
    monkeypatch.setattr(ds, "training_config", lambda b, a, log: SimpleNamespace(log_path=str(log)))
    mock = AsyncMock(side_effect=failure())
    monkeypatch.setattr(ds.train, "main", mock)
    with pytest.raises(failure):
        asyncio.run(ds.train_arm(tmp_path / "bundle", "random_success", tmp_path / "arm", "id"))
    with pytest.raises(ValueError):
        asyncio.run(ds.train_arm(tmp_path / "bundle", "random_success", tmp_path / "arm", "id"))
    assert mock.await_count == 1


@pytest.mark.parametrize("prior", ["claim", "unclaimed", "wrongidentity"])
def test_partial_train_no_implicit_resume(tmp_path, monkeypatch, prior):
    mock = AsyncMock()
    monkeypatch.setattr(ds.train, "main", mock)
    folder = tmp_path / "arm"
    claim = {
        "identity_sha256": "id",
        "arm": "random_success",
        "updates": 8,
        "epochs": 1,
        "fresh_base": True,
        "automatic_resume": False,
    }
    if prior == "unclaimed":
        write(folder / "training/checkpoints.jsonl", {})
    else:
        if prior == "wrongidentity":
            claim["identity_sha256"] = "wrong"
        write(folder / "claim.json", claim)
    with pytest.raises(ValueError):
        asyncio.run(ds.train_arm(tmp_path / "bundle", "random_success", folder, "id"))
    assert mock.await_count == 0


def test_training_config_fixes_budget(tmp_path):
    m, _, _ = dataset_fixture(tmp_path / "data")
    write(tmp_path / "bundle/manifest.json", {"arms": {"random_success": ds.pinned_file(m)}})
    cfg = ds.training_config(tmp_path / "bundle", "random_success", tmp_path / "log")
    assert (cfg.max_steps, cfg.num_epochs, cfg.lora_rank, cfg.learning_rate) == (8, 1, 32, 1e-5)
    assert cfg.load_checkpoint_path is None and cfg.save_every == 0 and cfg.eval_every == 0
    assert cfg.evaluator_builders == [] and cfg.submit_ahead == 1


@pytest.mark.parametrize("field", ["model_name", "max_turns", "split_seed", "learning_rate"])
def test_original_protocol_mismatch_rejected(field):
    config = {
        "model_name": ds.PROTOCOL["model_name"],
        "max_turns": 40,
        "train_samples": 4,
        "eval_samples": 4,
        "eval_size": 20,
        "split_seed": 7,
        "learning_rate": 1e-5,
    }
    evaluation = {
        "model_name": ds.PROTOCOL["model_name"],
        "max_turns": 40,
        "max_tokens": 16384,
        "max_sampled_tokens": 65536,
        "max_trajectory_tokens": 114688,
        "thinking_effort": 0.9,
        "temperature": 1.0,
    }
    ds.check_collection_protocol(config, evaluation)
    config[field] = "changed"
    with pytest.raises(ValueError):
        ds.check_collection_protocol(config, evaluation)


def test_bundle_changed_during_training_is_blocked(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    m = write(bundle / "manifest.json", {})
    proof = ds.pinned_file(m)
    monkeypatch.setattr(ds, "verify_bundle", lambda _: {})
    monkeypatch.setattr(ds, "training_config", lambda b, a, log: SimpleNamespace(log_path=str(log)))

    async def train(cfg):
        complete_training(Path(cfg.log_path))
        m.write_text('{"changed":true}')

    monkeypatch.setattr(ds.train, "main", AsyncMock(side_effect=train))
    with pytest.raises(ValueError):
        asyncio.run(ds.train_arm(bundle, "random_success", tmp_path / "arm", "id", proof))
    assert json.loads((tmp_path / "arm/result.json").read_text())["stage"] == "blocked"
