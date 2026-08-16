import json

from tinker_cookbook.recipes.kokkos_rl.dataset.release_manifest import build_manifest


def _write_trial(root, instance_id: str, agent: str, digest: str, reward: float) -> None:
    trial = root / f"{instance_id}__trial"
    trial.mkdir(parents=True)
    (trial / "lock.json").write_text(
        json.dumps({"task": {"name": instance_id, "digest": digest}})
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": f"org/{instance_id}",
                "agent_info": {"name": agent},
                "verifier_result": {"rewards": {"reward": reward}},
            }
        )
    )


def test_build_manifest_binds_results_to_task_digests(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.toml").write_text(
        """[dataset]
name = "org/data"
version = "2.0.0"

[[tasks]]
name = "org/old"
digest = "sha256:old"

[[tasks]]
name = "org/new"
digest = "sha256:new"
"""
    )
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "instances": [
                    {"instance_id": "old", "validation_source": "old.jsonl"},
                    {"instance_id": "new", "validation_source": "new.jsonl"},
                ]
            }
        )
    )
    old_instances = tmp_path / "old.jsonl"
    old_instances.write_text(json.dumps({"instance_id": "old"}) + "\n")
    evidence = tmp_path / "evidence"
    for instance_id, digest in (("old", "sha256:old"), ("new", "sha256:new")):
        _write_trial(evidence / "oracle", instance_id, "oracle", digest, 1.0)
        _write_trial(evidence / "nop", instance_id, "nop", digest, 0.0)

    manifest = build_manifest(dataset, old_instances, [evidence])

    assert manifest["instance_count"] == 2
    assert manifest["oracle_passed"] == 2
    assert manifest["nop_rejected"] == 2
    assert [item["release_group"] for item in manifest["instances"]] == [
        "v2-addition",
        "v1-unchanged",
    ]
