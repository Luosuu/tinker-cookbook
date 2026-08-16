import json

from tinker_cookbook.recipes.kokkos_rl.dataset.assemble import assemble_instances, write_dataset
from tinker_cookbook.recipes.kokkos_rl.dataset.models_test import _instance


def test_assemble_deduplicates_in_source_priority_order(tmp_path) -> None:
    instance = _instance()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(instance.to_dict()) + "\n")
    second.write_text(json.dumps(instance.to_dict()) + "\n")

    instances, provenance = assemble_instances([first, second])

    assert instances == [instance]
    assert provenance[instance.instance_id] == str(first)

    output = tmp_path / "dataset.jsonl"
    manifest = tmp_path / "manifest.json"
    write_dataset(instances, provenance, output, manifest)
    row = json.loads(output.read_text())
    assert row["instance_id"] == instance.instance_id
    assert row["FAIL_TO_PASS"] == ["Foo.Regression"]
    assert json.loads(manifest.read_text())["instance_count"] == 1
