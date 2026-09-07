import hashlib
import shutil
import subprocess
from dataclasses import replace

import pytest

from tinker_cookbook.recipes.kokkos_rl.dataset import test_assertions as assertions
from tinker_cookbook.recipes.kokkos_rl.dataset import test_commands
from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor_test import _local_instance


def synthetic_instance(tmp_path, monkeypatch):
    name = "kokkos__kokkos-8838"
    base, _, _, replacements = assertions.REPAIRS[name]
    original = "\n".join(before for before, _ in replacements)
    repaired = "\n".join(after for _, after in replacements)

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    monkeypatch.setattr(
        assertions, "REPAIRS", {name: (base, digest(original), digest(repaired), replacements)}
    )
    instance = replace(
        _local_instance(tmp_path)[1], instance_id=name, base_commit=base, test_patch=original
    )
    return instance, repaired


def test_repair_is_exact_idempotent_and_preserves_annotation_role_checks(tmp_path, monkeypatch):
    instance, expected = synthetic_instance(tmp_path, monkeypatch)
    entry = {
        "base_commit": instance.base_commit,
        "test_patch_sha256": hashlib.sha256(instance.test_patch.encode()).hexdigest(),
        "instance_fields": {
            "build_targets": {
                "original": list(instance.build_targets),
                "replacement": ["ReviewedTarget"],
            }
        },
    }
    monkeypatch.setattr(test_commands, "_reviewed_overrides", lambda: {instance.instance_id: entry})
    repaired = test_commands.apply_reviewed_test_overrides(instance)
    assert repaired.test_patch == expected
    assert repaired.build_targets == ("ReviewedTarget",)
    assert test_commands.apply_reviewed_test_overrides(repaired) == repaired
    assert repaired.code_patch == instance.code_patch
    assert repaired.test_patch.count("\n") == instance.test_patch.count("\n")
    with pytest.raises(ValueError, match="different hidden"):
        assertions.repair_assertions(replace(instance, test_patch=instance.test_patch + "changed"))


def test_other_base_or_task_is_never_rewritten(tmp_path, monkeypatch):
    instance, _ = synthetic_instance(tmp_path, monkeypatch)
    for other in [
        replace(instance, base_commit="another"),
        replace(instance, instance_id="another"),
    ]:
        assert assertions.repair_assertions(other)[0] == other


def test_repaired_cpp_expressions_reject_bad_stride_and_preserve_expected_values(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is needed to verify actual conditional precedence")
    lines = [
        "#include <cstddef>",
        "int main() {",
        "constexpr std::size_t KOKKOS_INVALID_INDEX = ~std::size_t{0};",
    ]
    for name, (_base, _old, _new, replacements) in assertions.REPAIRS.items():
        for index, (before, after) in enumerate(replacements):
            extent = 11 if index < (1 if name.endswith("8370") else 2) else 7
            right_invalid = " : KOKKOS_INVALID_INDEX" in before
            for is_ll in (False, True):
                for stride in (0, 1, 5, 7, 11, 99, -1):
                    literal = "KOKKOS_INVALID_INDEX" if stride == -1 else str(stride)
                    expected_value = extent if is_ll else -1 if right_invalid else 5
                    expected = stride == expected_value or (name.endswith("8370") and stride == -1)
                    lines += [
                        "{",
                        f"bool is_ll = {'true' if is_ll else 'false'};",
                        f"struct {{ std::size_t stride; }} l{{{literal}}};",
                        f"if (!({before})) return 1;",  # Confirms the original was vacuous.
                        f"if (bool({after}) != {'true' if expected else 'false'}) return 2;",
                        "}",
                    ]
    lines += ["return 0;", "}"]
    source, binary = tmp_path / "precedence.cpp", tmp_path / "precedence"
    source.write_text("\n".join(lines))
    subprocess.run(
        [compiler, "-std=c++17", "-w", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)
