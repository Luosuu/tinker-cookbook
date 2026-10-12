import argparse
import json
from types import SimpleNamespace

import pytest

from ..spend import BudgetStop
from . import scale_expansion
from .expansion import atomic_json, rows, write_rows
from .models import KokkosInstance
from .scale_expansion import profile_reason, shard_candidates, shard_of


def candidate(number: int, **metadata) -> dict:
    return {
        "instance_id": f"kokkos__kokkos-{number}",
        "repo": "kokkos/kokkos",
        "pr_number": number,
        "title": f"Fix {number}",
        "problem_statement": "",
        "merged_at": "2024-01-01T00:00:00Z",
        "base_commit": f"base{number}",
        "merge_commit": f"merge{number}",
        "era": "modern",
        "code_patch": f"diff --git a/core/src/F{number}.hpp b/core/src/F{number}.hpp\n+line {number}\n",
        "metadata": {"test_origin": "upstream", "accelerator": "cpu", **metadata},
    }


def test_shards_partition_and_deduplicate(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_rows(first, [candidate(n) for n in range(40)])
    write_rows(second, [candidate(n) for n in range(30, 60)])
    shards = [shard_candidates([first, second], s, 4) for s in range(4)]
    ids = [row["instance_id"] for shard in shards for row in shard]
    assert sorted(ids) == sorted({f"kokkos__kokkos-{n}" for n in range(60)})
    assert all(shard_of(row["instance_id"], 4) == s for s, rs in enumerate(shards) for row in rs)


def test_profile_gate():
    assert profile_reason(KokkosInstance.from_dict(candidate(1))) is None
    model_tests = candidate(2, test_origin="model")
    assert "test origin" in profile_reason(KokkosInstance.from_dict(model_tests))
    gpu = candidate(3, accelerator="cuda")
    assert "profile" in profile_reason(KokkosInstance.from_dict(gpu))
    kernels = {**candidate(4), "repo": "kokkos/kokkos-kernels"}
    assert "dependency" in profile_reason(KokkosInstance.from_dict(kernels))
    kernels["metadata"]["kokkos_dependency_commit"] = "abc"
    assert profile_reason(KokkosInstance.from_dict(kernels)) is None


def test_cstdint_compatibility_only_for_that_build_error():
    from .auto_annotate import _normalize_toolchain_annotation
    from .scale_expansion import needs_cstdint, with_cstdint

    def preflight(stage, text):
        return {"preflight": {"stage": stage, "records": [{"stdout": text, "stderr": ""}]}}

    error = "Kokkos_MemoryPool.cpp:112:48: error: ‘uint32_t’ has not been declared"
    assert needs_cstdint(preflight("build", error))
    assert not needs_cstdint(preflight("build", "error: other"))
    assert not needs_cstdint(preflight("target_resolution", error))
    instance = with_cstdint(KokkosInstance.from_dict(candidate(1)))
    assert instance.configure_command.endswith('-DCMAKE_CXX_FLAGS="-include cstdint"')
    annotation = {"configure_command": "cmake -S . -B build", "metadata": {"toolchain": "cpu"}}
    normalized = _normalize_toolchain_annotation(annotation, instance)
    assert normalized["configure_command"].count("-include cstdint") == 1


class FakeCompleter:
    def __init__(self, **_):
        create = None
        self.client = SimpleNamespace(
            with_options=lambda **_: self.client,
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

    async def close(self):
        pass


def arguments(tmp_path, candidates_path) -> argparse.Namespace:
    frozen = tmp_path / "frozen.jsonl"
    write_rows(frozen, [candidate(999)])
    return argparse.Namespace(
        candidates=[candidates_path],
        frozen_instances=frozen,
        prior_accepted=[],
        output_dir=tmp_path / "campaign",
        shard=0,
        shards=1,
        rivanna_host="host",
        rivanna_root="/scratch/run",
        rivanna_base_sif="/scratch/base.sif",
        rivanna_partition="bii-largemem",
        base_url="https://example.invalid",
        model="model",
        api_key_env="TINKER_API_KEY",
        env_file=tmp_path / "missing.env",
        provider_parameters={},
        max_tokens=16,
        max_attempts=1,
        max_statement_revisions=1,
        ledger=tmp_path / "campaign" / "spend.json",
        max_usd=1.0,
    )


@pytest.mark.asyncio
async def test_run_records_outcomes_and_stops_on_budget(tmp_path, monkeypatch):
    from .. import chat_inference
    from . import rivanna_bridge

    monkeypatch.setattr(chat_inference, "ChatAnnotationCompleter", FakeCompleter)
    monkeypatch.setattr(rivanna_bridge, "RivannaBridge", lambda *a, **k: None)
    monkeypatch.setenv("FAKE_API_KEY", "secret-value-123")
    candidates_path = tmp_path / "candidates.jsonl"
    write_rows(candidates_path, [candidate(n) for n in (1, 2, 999, 3)])
    calls = []

    async def fake_process(instance, *, directory, **_):
        calls.append(instance.pr_number)
        if instance.pr_number == 2:
            atomic_json(directory / "progress.json", {"stage": "construction"})
            raise ValueError("Assertion mismatch secret-value-123")
        if instance.pr_number == 3:
            raise BudgetStop("cap")
        atomic_json(directory / "validated.json", instance.to_dict())
        return {"export_sha256": "e", "trials": [], "validated": instance}

    monkeypatch.setattr(scale_expansion, "process", fake_process)
    args = arguments(tmp_path, candidates_path)
    await scale_expansion.run(args)

    shard = args.output_dir / "shard-00"
    report = json.loads((shard / "report.json").read_text())
    assert report["status"].startswith("budget_stop")
    assert report["processed"] == 3 and report["accepted"] == 1
    assert calls == [1, 2, 3]
    failed = json.loads((shard / "kokkos__kokkos-2" / "receipt.json").read_text())
    assert failed["stage"] == "construction"
    assert "secret-value-123" not in failed["reason"]
    frozen = json.loads((shard / "kokkos__kokkos-999" / "receipt.json").read_text())
    assert frozen["reason"].startswith("frozen-family")
    assert not (shard / "kokkos__kokkos-3" / "receipt.json").exists()

    calls.clear()

    async def finish(instance, *, directory, **_):
        calls.append(instance.pr_number)
        atomic_json(directory / "validated.json", instance.to_dict())
        return {"export_sha256": "e", "trials": [], "validated": instance}

    monkeypatch.setattr(scale_expansion, "process", finish)
    await scale_expansion.run(args)
    assert calls == [3]
    assert json.loads((shard / "report.json").read_text())["status"] == "complete"
    assert [row["pr_number"] for row in rows(shard / "accepted_training.jsonl")] == [1, 3]


@pytest.mark.asyncio
async def test_rivanna_outage_retries_then_stops_without_receipt(tmp_path, monkeypatch):
    from .. import chat_inference
    from . import rivanna_bridge

    monkeypatch.setattr(chat_inference, "ChatAnnotationCompleter", FakeCompleter)
    monkeypatch.setattr(rivanna_bridge, "RivannaBridge", lambda *a, **k: None)
    monkeypatch.setattr(scale_expansion, "OUTAGE_RETRY_SECONDS", 0)
    candidates_path = tmp_path / "candidates.jsonl"
    write_rows(candidates_path, [candidate(n) for n in (1, 2)])
    outages = {1: 2, 2: 99}

    async def flaky(instance, *, directory, **_):
        if outages[instance.pr_number]:
            outages[instance.pr_number] -= 1
            atomic_json(directory / "construction.json", {"cached": "failure"})
            raise ValueError("RuntimeError: ssh failed: ")
        assert not (directory / "construction.json").exists()
        atomic_json(directory / "validated.json", instance.to_dict())
        return {"export_sha256": "e", "trials": [], "validated": instance}

    monkeypatch.setattr(scale_expansion, "process", flaky)
    args = arguments(tmp_path, candidates_path)
    await scale_expansion.run(args)
    shard = args.output_dir / "shard-00"
    assert json.loads((shard / "kokkos__kokkos-1" / "receipt.json").read_text())["status"] == (
        "accepted"
    )
    assert not (shard / "kokkos__kokkos-2" / "receipt.json").exists()
    assert json.loads((shard / "report.json").read_text())["status"].startswith(
        "infrastructure_stop"
    )
