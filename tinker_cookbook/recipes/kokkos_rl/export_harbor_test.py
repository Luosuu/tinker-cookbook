import json
import tomllib
from dataclasses import replace

from tinker_cookbook.recipes.kokkos_rl.export_harbor import (
    _agent_allowed_code_patch,
    export_instance,
)
from tinker_cookbook.recipes.kokkos_rl.models_test import _instance


def test_export_hides_tests_and_adds_clean_room_guard(tmp_path) -> None:
    instance = _instance()
    task_dir = export_instance(instance, tmp_path)

    assert (task_dir / "environment" / "Dockerfile").is_file()
    assert "ca-certificates" in (task_dir / "environment" / "Dockerfile").read_text()
    assert (task_dir / "tests" / "test.patch").read_text() == "tests"
    assert (task_dir / "solution" / "gold.patch").read_text() == "code"
    assert "git apply" in (task_dir / "solution" / "solve.sh").read_text()
    assert "Foo.Regression" not in (task_dir / "instruction.md").read_text()

    test_script = (task_dir / "tests" / "test.sh").read_text()
    assert "git diff --name-only" in test_script
    assert "git ls-files --others" in test_script
    assert r"sed '\#^build/#d'" in test_script
    assert "git apply --whitespace=nowarn /tests/test.patch" in test_script
    assert "echo 1" in test_script

    metadata = json.loads((task_dir / "metadata.json").read_text())
    assert metadata["instance_id"] == instance.instance_id

    task_config = tomllib.loads((task_dir / "task.toml").read_text())
    assert task_config["schema_version"] == "1.4"
    assert task_config["task"]["name"] == f"swe-kokkos/{instance.instance_id}"
    assert task_config["environment"]["network_mode"] == "public"
    assert task_config["agent"]["network_mode"] == "allowlist"
    assert task_config["agent"]["allowed_hosts"] == ["api.openai.com"]
    assert task_config["verifier"]["network_mode"] == "no-network"


def test_oracle_patch_excludes_verifier_protected_paths() -> None:
    instance = replace(
        _instance(),
        code_patch="""diff --git a/core/src/Foo.hpp b/core/src/Foo.hpp
--- a/core/src/Foo.hpp
+++ b/core/src/Foo.hpp
@@ -1 +1 @@
-old
+new
diff --git a/example/CMakeLists.txt b/example/CMakeLists.txt
--- a/example/CMakeLists.txt
+++ b/example/CMakeLists.txt
@@ -1 +1 @@
-old
+new
""",
    )

    allowed = _agent_allowed_code_patch(instance)

    assert "core/src/Foo.hpp" in allowed
    assert "CMakeLists.txt" not in allowed
