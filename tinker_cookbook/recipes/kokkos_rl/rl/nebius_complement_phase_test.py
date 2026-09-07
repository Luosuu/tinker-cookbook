import json

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_nebius import MODELS
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import CLIConfig
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_complement_phase import (
    evaluation_config,
    require_complete_review,
    validate_catalog,
)
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import pinned_file
from tinker_cookbook.utils.ml_log import dump_config


def original_config():
    return dump_config(
        CLIConfig(
            model_name=MODELS[0],
            api_mode="chat",
            chat_provider="nebius",
            reasoning_effort="high",
            temperature=None,
            max_turns=40,
            max_tokens=16384,
            max_sampled_tokens=65536,
            max_input_tokens=5_000_000,
            max_tool_calls=80,
            grader_timeout=900,
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_turns", 41),
        ("max_tokens", 8192),
        ("reasoning_effort", "low"),
        ("chat_provider", "tinker"),
        ("allow_network", True),
    ],
)
def test_changed_sampling_protocol_is_rejected(field, value):
    saved = original_config()
    saved[field] = value
    with pytest.raises(ValueError, match="sampling protocol"):
        evaluation_config(saved, MODELS[0], {})


def test_execution_policy_is_in_each_model_identity_without_expanding_budget():
    policy = {
        "resources": {"task": {"backend": "modal", "gpu": "L4"}},
        "network_policy": {"sdk_retries": 0},
        "required_environment_policies": {"task": "env1"},
    }
    result = evaluation_config(original_config(), MODELS[0], policy)
    assert result.sandbox_resource_policy is not None
    assert json.loads(result.sandbox_resource_policy) == policy
    assert result.max_turns == 40 and result.max_sampled_tokens == 65536
    assert result.max_concurrency == 1 and result.max_infra_retries == 0


def frozen_review(tmp_path):
    hashes = {f"task-{i}": f"digest-{i}" for i in range(100)}
    qualification = tmp_path / "qualification.json"
    coverage = tmp_path / "coverage.json"
    approval = tmp_path / "approval.json"
    source = tmp_path / "source_review.json"
    source.write_text('{"explicit_review":true}')
    qualification.write_text(
        json.dumps({"task_hashes": hashes, "qualified": 100, "ready_for_sampling": True})
    )
    coverage.write_text(json.dumps({"task_hashes": hashes, "status": "complete", "blockers": []}))
    approval.write_text(
        json.dumps(
            {
                "approvals": {
                    n: {
                        "task": n,
                        "task_hash": d,
                        "status": "accepted",
                        "sources": [pinned_file(source)],
                    }
                    for n, d in hashes.items()
                }
            }
        )
    )
    return hashes, qualification, coverage, approval, source


def test_complete_report_still_requires_every_explicit_scope_approval(tmp_path):
    hashes, qualification, coverage, approval, _ = frozen_review(tmp_path)
    require_complete_review(hashes, qualification, coverage, approval)
    data = json.loads(approval.read_text())
    del data["approvals"]["task-99"]
    approval.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        require_complete_review(hashes, qualification, coverage, approval)


@pytest.mark.parametrize("mutate", ["scope_source", "hash", "pending"])
def test_stale_or_incomplete_review_cannot_prepare_a_phase(tmp_path, mutate):
    hashes, qualification, coverage, approval, source = frozen_review(tmp_path)
    if mutate == "scope_source":
        source.write_text("changed")
    elif mutate == "hash":
        hashes["task-0"] = "changed"
    else:
        data = json.loads(coverage.read_text())
        data["status"] = "pending"
        coverage.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        require_complete_review(hashes, qualification, coverage, approval)


def test_catalog_must_preserve_exact_models_and_pricing():
    configs = {
        m: CLIConfig(input_price_per_million=0.15, output_price_per_million=0.5) for m in MODELS
    }
    rows = [
        {"id": m, "pricing": {"prompt": "0.00000015", "completion": "0.0000005"}} for m in MODELS
    ]
    validate_catalog({"data": rows}, configs)
    rows[0]["pricing"]["prompt"] = "0.0000015"
    with pytest.raises(ValueError, match="pricing changed"):
        validate_catalog({"data": rows}, configs)
    with pytest.raises(ValueError, match="exact requested model"):
        validate_catalog({"data": rows[1:]}, configs)


@pytest.mark.asyncio
async def test_full_dry_run_reserves_original_queue_without_clients_or_claims(
    tmp_path, monkeypatch
):
    from dataclasses import asdict
    from datetime import UTC, datetime

    from tinker_cookbook.recipes.kokkos_rl.rl import nebius_complement_phase as phase
    from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger_test import write
    from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot import Policy
    from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot_test import task_with_metadata

    original, snapshot, output = tmp_path / "original", tmp_path / "snapshot", tmp_path / "phase"
    tasks = [task_with_metadata(snapshot / "tasks", f"task-{i:03d}") for i in range(100)]
    for task in tasks:
        (task.task_dir / "instruction.md").write_text(task.instruction)
        (task.task_dir / "task.toml").write_text("")
    hashes = {t.task_name: phase.current_task_digest(t) for t in tasks}
    write(snapshot / "manifest.json", {"task_hashes": hashes})
    manifest = write(original / "manifest.json", {"task_hashes": hashes})
    write(
        original / "launch.json",
        {
            "source_manifest": {"task_hashes": hashes},
            "config": {
                "source_manifest": str(manifest),
                "max_concurrency": 4,
                "modal_task_names": [],
                "validation_dir": str(original / "validation"),
            },
        },
    )
    write(original / "validation/task-000.json", {"passed": True})
    for model in MODELS:
        saved = original_config()
        saved["model_name"] = model
        write(
            original / model.split("/")[-1] / "eval_identity.json",
            {"tasks": hashes, "identity": {"config": saved}},
        )
    # Three unknown original requests occupy slots; the fourth reserved pair
    # has not started, so none of the model capacity can be borrowed yet.
    for model in MODELS[:3]:
        write(original / model.split("/")[-1] / "task-000/attempt_started.json", {})
    write(
        original / "status.json",
        {
            "updated_at": datetime.now(UTC).isoformat(),
            "active": [[m, "task-000"] for m in MODELS[:3]],
        },
    )
    source = write(tmp_path / "manual_scope.json", {"review": "accepted configured scope"})
    approvals = write(
        tmp_path / "scope.json",
        {
            "approvals": {
                n: {
                    "task": n,
                    "task_hash": d,
                    "status": "accepted",
                    "sources": [pinned_file(source)],
                }
                for n, d in hashes.items()
            }
        },
    )
    coverage = write(
        tmp_path / "coverage.json", {"task_hashes": hashes, "status": "complete", "blockers": []}
    )
    qrows = {}
    for n, d in hashes.items():
        evidence = write(
            tmp_path / "evidence" / f"{n}.json",
            {"task": n, "task_hash": d, **asdict(Policy()), "nop": 0, "oracle": 1, "passed": True},
        )
        qrows[n] = {"qualified": True, "successful_evidence": [pinned_file(evidence)]}
    qualification = write(
        tmp_path / "qualification.json",
        {"task_hashes": hashes, "tasks": qrows, "ready_for_sampling": True, "qualified": 100},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not initialize remote work or claim a sample")

    for name in ["create_nebius_client", "ImportedCacheFactory", "claim_pair", "evaluate_task"]:
        monkeypatch.setattr(phase, name, forbidden)
    await phase.main(
        phase.Config(
            original_root=str(original),
            output_path=str(output),
            snapshot_dir=str(snapshot),
            qualification_path=str(qualification),
            coverage_path=str(coverage),
            scope_approval_path=str(approvals),
            required_environment_policies=(),
            dispatch=False,
        )
    )
    report = json.loads((output / "preflight.json").read_text())
    assert len(report["pending_pairs"]) == 396
    assert report["capacity"]["available_model_slots"] == 0
    assert report["capacity"]["reserved_unstarted_pairs"] == 1
    assert report["new_claims"] == report["new_model_requests"] == 0
    assert not list(original.rglob("claim.json"))

    second = tmp_path / "second-phase"
    await phase.main(
        phase.Config(
            original_root=str(original),
            output_path=str(second),
            snapshot_dir=str(snapshot),
            qualification_path=str(qualification),
            coverage_path=str(coverage),
            scope_approval_path=str(approvals),
            required_environment_policies=(),
            dispatch=False,
        )
    )
    assert (
        pinned_file(output / "phase_identity.json")["sha256"]
        != pinned_file(second / "phase_identity.json")["sha256"]
    )
