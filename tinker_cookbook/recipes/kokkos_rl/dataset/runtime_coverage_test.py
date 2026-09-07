import shlex
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor_test import (
    _local_instance,
    _run_verifier,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.runtime_coverage import (
    COVERAGE_FAILURE,
    coverage_error,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.test_commands import (
    guarded_command,
    normalize_filter,
    normalize_test_command,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.validate import _run
from tinker_cookbook.sandbox import SandboxInterface, SandboxResult


def test_filters_preserve_all_positive_and_negative_cases_and_require_rename_evidence():
    assert (
        normalize_filter("TEST_CATEGORY.one:TEST_CATEGORY.two-TEST_CATEGORY_DEATH.three")
        == "*.one:*.two-*.three"
    )
    assert normalize_filter("Suite.exact") == "Suite.exact"
    command = "./binary --gtest_filter=space_aware_accessor*"
    patch = "-TEST(TEST_CATEGORY, space_aware_accessor) {}\n+TEST(TEST_CATEGORY, mdspan_space_aware_accessor) {}\n"
    assert "*.mdspan_space_aware_accessor*" in normalize_test_command(command, test_patch=patch)
    assert "*.space_aware_accessor*" in normalize_test_command(
        command, test_patch=patch, after_test_patch=False
    )
    assert "mdspan" not in normalize_test_command(command)
    ambiguous = patch + "+TEST(TEST_CATEGORY, another_space_aware_accessor) {}\n"
    assert "mdspan" not in normalize_test_command(command, test_patch=ambiguous)
    assert (
        normalize_test_command("./binary '--gtest_filter=*.deep_copy_same_view'")
        == "./binary --gtest_filter='*.deep_copy_same_view'"
    )
    assert (
        normalize_test_command(r"./binary --gtest_filter=\'*Scatter*\'")
        == "./binary --gtest_filter='*Scatter*'"
    )


@pytest.mark.parametrize(
    "output",
    [
        "[==========] Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests.",
        "[==========] Running 1 test from 1 test suite.\n[  PASSED  ] 1 test.\n[==========] Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests.",
        "[==========] Running 1 test from 1 test suite.\n[  PASSED  ] 0 tests.\n[ SKIPPED ] 1 test.",
        "unknown flag, showing help",
    ],
)
def test_zero_missing_and_all_skipped_runtime_execution_are_not_success(output):
    command = (
        "python3 -c " + shlex.quote("print(" + repr(output) + ")") + " --gtest_filter=Suite.case"
    )
    result = subprocess.run(guarded_command(command), shell=True, text=True, capture_output=True)
    assert result.returncode == COVERAGE_FAILURE
    assert "KOKKOS_RUNTIME_COVERAGE_ERROR" in result.stderr


def test_ctest_zero_and_nested_zero_are_rejected_and_verbose_is_enabled():
    assert coverage_error("ctest -R missing", "No tests were found!!!", 0)
    assert coverage_error(
        "ctest -R empty",
        "1: [==========] Running 0 tests from 0 test suites.\n100% tests passed, 0 tests failed out of 1",
        0,
    )
    assert coverage_error("ctest -R real", "100% tests passed, 0 tests failed out of 1", 0) is None
    assert coverage_error("ctest -R failure | cat", "50% tests passed, 1 tests failed out of 2", 0)
    assert (
        normalize_test_command("cmake --build build && ctest -R real")
        == "cmake --build build && ctest --verbose --test-dir build -R real"
    )
    assert (
        normalize_test_command("ctest --test-dir build -R real")
        == "ctest --verbose --test-dir build -R real"
    )


def test_runtime_failure_needs_nonzero_selection_to_qualify_as_f2p(tmp_path):
    empty = (
        "python3 -c "
        + shlex.quote(
            "print('[==========] Running 0 tests from 0 test suites.'); raise SystemExit(1)"
        )
        + " --gtest_filter=Suite.missing"
    )
    result = _run(empty, cwd=tmp_path, timeout=10, expected_exit="nonzero", runtime_check=True)
    assert not result.matched_expectation and result.coverage_error
    real = (
        "python3 -c "
        + shlex.quote(
            "print('[==========] Running 1 test from 1 test suite.'); raise SystemExit(1)"
        )
        + " --gtest_filter=Suite.failing"
    )
    result = _run(real, cwd=tmp_path, timeout=10, expected_exit="nonzero", runtime_check=True)
    assert result.matched_expectation


@pytest.mark.asyncio
async def test_sandbox_validator_rejects_zero_tests_as_an_expected_failure(tmp_path):
    from tinker_cookbook.recipes.kokkos_rl.dataset import modal_validate

    async def execute(command, **kwargs):
        result = subprocess.run(
            ["bash", "-lc", command], cwd=tmp_path, text=True, capture_output=True
        )
        return SandboxResult(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode
        )

    sandbox = cast(SandboxInterface, SimpleNamespace(run_command=AsyncMock(side_effect=execute)))
    command = (
        "python3 -c "
        + shlex.quote(
            "print('[==========] Running 0 tests from 0 test suites.'); raise SystemExit(1)"
        )
        + " --gtest_filter=Suite.missing"
    )
    result = await modal_validate._run(
        sandbox, command, timeout=10, expected_exit="nonzero", runtime_check=True
    )
    assert not result.matched_expectation and result.coverage_error


@pytest.mark.parametrize("stage,selected", [("build", 0), ("test", 0), ("build", 1), ("test", 1)])
def test_exported_verifier_checks_runtime_p2p_even_for_compile_failure_tasks(
    tmp_path, stage, selected
):
    repo, instance = _local_instance(tmp_path)
    (repo / "src.txt").write_text("fixed")
    fake = tmp_path / "fake_runtime.py"
    fake.write_text(
        "import sys\nassert sys.argv[1] == '--gtest_filter=*.case'\n"
        + f"print('[==========] Running {selected} tests from 1 test suite.')\nprint('[  PASSED  ] {selected} tests.')\n"
    )
    instance = replace(
        instance,
        metadata={"f2p_stage": stage},
        p2p_commands=(f"python3 {shlex.quote(str(fake))} --gtest_filter=TEST_CATEGORY.case",),
    )
    assert _run_verifier(tmp_path, repo, instance) == str(int(selected > 0))


def test_reviewed_selectors_are_instance_and_base_scoped_and_idempotent(tmp_path):
    import json
    from pathlib import Path

    from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor import _test_script

    _, instance = _local_instance(tmp_path)
    overrides = json.loads(Path(__file__).with_name("test_command_overrides.json").read_text())
    expected = {
        "kokkos__kokkos-7428": "serial.task_*",
        "kokkos__kokkos-8594": "serial.scatterview:serial.scatterview_devicetype",
        "kokkos__kokkos-8967": "serial_DeathTest.view_subview_constructor_layout_compatibility",
    }
    for task, selector in expected.items():
        entry = overrides[task]
        old = next(iter(entry["selectors"]))
        pinned = replace(instance, instance_id=task, base_commit=entry["base_commit"])
        command = "./binary --gtest_filter=" + shlex.quote(old)
        actual = normalize_test_command(command, instance=pinned)
        assert shlex.split(actual)[1] == "--gtest_filter=" + selector
        assert normalize_test_command(actual, instance=pinned) == actual
        for unrelated in (
            replace(pinned, instance_id="another-instance"),
            replace(pinned, base_commit="0" * 40),
        ):
            assert normalize_test_command(command, instance=unrelated) == command
        script = _test_script(replace(pinned, p2p_commands=(command,)))
        assert selector in script
        assert old not in script
        assert all(item["url"].endswith(item["path"]) for item in entry["evidence"])
