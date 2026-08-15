import json

from tinker_cookbook.recipes.kokkos_rl.export_harbor import export_instance
from tinker_cookbook.recipes.kokkos_rl.models_test import _instance


def test_export_hides_tests_and_adds_clean_room_guard(tmp_path) -> None:
    instance = _instance()
    task_dir = export_instance(instance, tmp_path)

    assert (task_dir / "environment" / "Dockerfile").is_file()
    assert "ca-certificates" in (task_dir / "environment" / "Dockerfile").read_text()
    assert (task_dir / "tests" / "test.patch").read_text() == "tests"
    assert (task_dir / "solution" / "gold.patch").read_text() == "full"
    assert "Foo.Regression" not in (task_dir / "instruction.md").read_text()

    test_script = (task_dir / "tests" / "test.sh").read_text()
    assert "git diff --name-only" in test_script
    assert "git apply --whitespace=nowarn /tests/test.patch" in test_script
    assert "echo 1" in test_script

    metadata = json.loads((task_dir / "metadata.json").read_text())
    assert metadata["instance_id"] == instance.instance_id
