import json
import os

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import original_gate
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger_test import write
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_retire_gate import plan_retirement, retire_planned


def fixture(tmp_path):
    original = tmp_path / "original"
    manifest = write(
        original / "source.json", {"task_hashes": {"normal": "n", "modal": "m", "failed": "f"}}
    )
    write(
        original / "launch.json",
        {
            "config": {
                "source_manifest": str(manifest),
                "validation_dir": str(original / "validation"),
                "modal_task_names": ["modal"],
                "modal_memory_mb": 16384,
                "modal_cpus": 4,
            }
        },
    )
    normal = write(
        original / "validation/normal.json", {"passed": True, "verifier": "original exact bytes"}
    )
    modal = write(
        original / "validation_overrides/modal.json",
        {"passed": True, "backend": "modal", "task_hash": "m", "memory_mb": 16384, "cpu": 4},
    )
    failed = write(original / "validation/failed.json", {"passed": False})
    attempt = write(original / "GLM-5.3-Flash/normal/attempt_started.json", {"unknown": True})
    return original, normal, modal, failed, attempt


def test_plan_precedes_moves_and_preserves_every_original_byte(tmp_path):
    original, normal, modal, failed, attempt = fixture(tmp_path)
    before = {p: p.read_bytes() for p in [normal, modal, failed, attempt]}
    holding = tmp_path / "holding"
    manifest = plan_retirement(original, holding)
    assert all(p.read_bytes() == data for p, data in before.items())
    retire_planned(manifest)
    assert original_gate(original)[0] == frozenset()
    assert not normal.exists() and not modal.exists()
    assert (holding / "records/normal/normal.json").read_bytes() == before[normal]
    assert (holding / "records/modal/modal.json").read_bytes() == before[modal]
    assert failed.read_bytes() == before[failed] and attempt.read_bytes() == before[attempt]
    assert len((holding / "move_log.jsonl").read_text().splitlines()) == 2
    retire_planned(manifest)
    assert failed.read_bytes() == before[failed] and attempt.read_bytes() == before[attempt]


def test_resume_after_link_before_unlink_is_lossless(tmp_path):
    original, normal, _, _, _ = fixture(tmp_path)
    manifest = plan_retirement(original, tmp_path / "holding")
    entry = next(
        row
        for row in json.loads(manifest.read_text())["entries"]
        if row["source"]["path"] == str(normal)
    )
    destination = type(normal)(entry["destination"])
    destination.parent.mkdir(parents=True)
    os.link(normal, destination)
    retire_planned(manifest)
    assert destination.exists() and not normal.exists()
    assert original_gate(original)[0] == frozenset()


def test_changed_record_blocks_before_touching_that_source(tmp_path):
    original, normal, _, _, _ = fixture(tmp_path)
    manifest = plan_retirement(original, tmp_path / "holding")
    normal.write_text('{"passed":true,"changed":true}')
    with pytest.raises(ValueError, match="record changed"):
        retire_planned(manifest)
    assert normal.exists()


def test_new_permission_during_retirement_blocks_capacity_transition(tmp_path):
    original, _, _, failed, _ = fixture(tmp_path)
    manifest = plan_retirement(original, tmp_path / "holding")
    failed.write_text('{"passed":true}')
    with pytest.raises(ValueError, match="New legacy permissions"):
        retire_planned(manifest)
    assert failed.exists()
    assert original_gate(original)[0] == frozenset({"failed"})
