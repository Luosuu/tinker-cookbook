import json
from dataclasses import replace

import pytest

from tinker_cookbook.recipes.kokkos_rl.dataset.annotate import apply_annotations
from tinker_cookbook.recipes.kokkos_rl.dataset.auto_annotate import (
    _infer_core_target_from_context,
    _normalize_toolchain_annotation,
    annotate_and_validate_instance,
    parse_annotation,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.models import ChangedFile
from tinker_cookbook.recipes.kokkos_rl.dataset.models_test import _instance
from tinker_cookbook.renderers import Message
from tinker_cookbook.sandbox import SandboxResult


def _annotation(target: str) -> str:
    return json.dumps(
        {
            "instance_id": "kokkos__kokkos-1",
            "build_targets": [target],
            "fail_to_pass": [target],
            "pass_to_pass": [],
            "f2p_commands": [],
            "p2p_commands": [],
            "metadata": {"f2p_stage": "build"},
        }
    )


class _FakeCompleter:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []

    async def __call__(self, messages: list[Message]) -> Message:
        self.prompts.append(str(messages[-1]["content"]))
        return {"role": "assistant", "content": next(self.outputs)}


class _FakeSandbox:
    def __init__(self, target: str) -> None:
        self.target = target
        self.build_calls = 0
        self.cleaned = False

    @property
    def sandbox_id(self) -> str:
        return "fake"

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        if command.startswith("cmake --build"):
            self.build_calls += 1
            if self.target.endswith("BadTarget"):
                return SandboxResult(stdout="", stderr="unknown target BadTarget", exit_code=1)
            exit_code = 1 if self.build_calls == 2 else 0
            return SandboxResult(stdout="build", stderr="", exit_code=exit_code)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        self.cleaned = True


def test_parse_annotation_accepts_fenced_json() -> None:
    parsed = parse_annotation(f"```json\n{_annotation('GoodTarget')}\n```", "kokkos__kokkos-1")
    assert parsed["build_targets"] == ["GoodTarget"]
    assert parsed["metadata"] == {
        "f2p_stage": "build",
        "requires_gpu": False,
        "toolchain": "cpu",
    }


def test_parse_annotation_rejects_test_stage_without_command() -> None:
    value = json.loads(_annotation("GoodTarget"))
    value["metadata"]["f2p_stage"] = "test"
    with pytest.raises(ValueError, match="at least one f2p command"):
        parse_annotation(json.dumps(value), "kokkos__kokkos-1")


def test_apply_annotations_rejects_candidate_field_overwrite() -> None:
    with pytest.raises(ValueError, match="unsupported annotation fields"):
        apply_annotations(_instance(), {"base_commit": "malicious"})


def test_cuda_toolchain_configuration_is_normalized() -> None:
    annotation = {
        "configure_command": (
            "cmake -S . -B build -DKokkos_ENABLE_CUDA=ON "
            "-DKokkos_ENABLE_OPENMP=ON -DCMAKE_CXX_COMPILER=nvcc "
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        ),
        "metadata": {"toolchain": "cuda"},
    }
    normalized = _normalize_toolchain_annotation(annotation, _instance())
    command = str(normalized["configure_command"])
    assert "bin/nvcc_wrapper" in command
    assert "Kokkos_ARCH_ADA89=ON" in command
    assert "Kokkos_ENABLE_OPENMP=OFF" in command
    assert "CMAKE_CXX_COMPILER_LAUNCHER" not in command


def test_recent_core_build_failure_uses_all_cpu_aggregate_targets() -> None:
    annotation = {
        "build_command": "",
        "build_targets": ["InventedTarget"],
        "p2p_commands": ["ctest --test-dir build -R TooBroad"],
        "pass_to_pass": ["TooBroad"],
        "metadata": {"f2p_stage": "build", "toolchain": "cpu"},
    }
    instance = replace(
        _instance(),
        changed_files=(ChangedFile("core/unit_test/TestComplex.hpp", "modified", 3, 2),),
    )

    normalized = _normalize_toolchain_annotation(annotation, instance, attempt_number=3)

    assert normalized["build_command"] == ""
    assert normalized["build_targets"] == [
        "Kokkos_CoreTestCompileOnly",
        "Kokkos_CoreUnitTest_Serial1",
        "Kokkos_CoreUnitTest_Serial2",
    ]
    assert normalized["p2p_commands"] == ["ctest --verbose --test-dir build -R TooBroad"]
    assert normalized["pass_to_pass"] == ["TooBroad"]


def test_core_target_is_inferred_from_cmake_source_list() -> None:
    instance = replace(
        _instance(),
        changed_files=(ChangedFile("core/unit_test/TestQuadPrecisionMath.hpp", "modified", 3, 2),),
    )
    context = """# core/unit_test/CMakeLists.txt
153: set(${Tag}_TESTNAMES1B
172:   QuadPrecisionMath
519: kokkos_add_executable_and_test(CoreUnitTest_Serial1 SOURCES ${Serial_SOURCES1})
"""

    assert _infer_core_target_from_context(instance, context) == ("Kokkos_CoreUnitTest_Serial1")
    annotation = {
        "build_targets": ["CoreUnitTest_Serial2"],
        "metadata": {"f2p_stage": "test", "toolchain": "cpu"},
    }
    normalized = _normalize_toolchain_annotation(
        annotation,
        instance,
        build_context=context,
    )
    assert normalized["build_targets"] == ["Kokkos_CoreUnitTest_Serial1"]


def test_kokkos_macro_target_gets_project_prefix() -> None:
    annotation = {
        "build_command": "",
        "build_targets": ["ContainersUnitTest_Serial"],
        "metadata": {"f2p_stage": "test", "toolchain": "cpu"},
    }

    normalized = _normalize_toolchain_annotation(annotation, _instance())

    assert normalized["build_targets"] == ["Kokkos_ContainersUnitTest_Serial"]


def test_gtest_case_is_not_passed_to_ctest_regex() -> None:
    annotation = {
        "build_command": "",
        "build_targets": ["CoreUnitTest_Serial1"],
        "fail_to_pass": ["TestSerial_Math/new_regression"],
        "pass_to_pass": ["TestSerial_Math/existing_case"],
        "f2p_commands": ["ctest --test-dir build -R new_regression"],
        "p2p_commands": ["ctest --test-dir build -R existing_case"],
        "metadata": {"f2p_stage": "test", "toolchain": "cpu"},
    }

    normalized = _normalize_toolchain_annotation(annotation, _instance())

    assert normalized["f2p_commands"] == [
        '"$(find build -type f -name Kokkos_CoreUnitTest_Serial1 '
        "-perm -111 -print -quit)\" --gtest_filter='*new_regression*'"
    ]
    assert normalized["p2p_commands"] == [
        '"$(find build -type f -name Kokkos_CoreUnitTest_Serial1 '
        "-perm -111 -print -quit)\" --gtest_filter='*existing_case*'"
    ]


def test_existing_gtest_command_uses_discovered_binary_path() -> None:
    annotation = {
        "build_targets": ["CoreUnitTest_InitializeFinalize"],
        "f2p_commands": [
            "./build/core/unit_test/Kokkos_CoreUnitTest_InitializeFinalize "
            "--gtest_filter=Suite.regression"
        ],
        "p2p_commands": [],
        "metadata": {"f2p_stage": "test", "toolchain": "cpu"},
    }

    normalized = _normalize_toolchain_annotation(annotation, _instance())

    assert normalized["f2p_commands"] == [
        '"$(find build -type f -name Kokkos_CoreUnitTest_InitializeFinalize '
        '-perm -111 -print -quit)" --gtest_filter=Suite.regression'
    ]


def test_test_stage_removes_build_command_and_expands_category_macro() -> None:
    annotation = {
        "build_targets": ["CoreUnitTest_Serial1"],
        "f2p_commands": [
            "cmake --build build --target Kokkos_CoreUnitTest_Serial1",
            "./wrong/path --gtest_filter=TEST_CATEGORY.numeric_traits_denorm_min",
        ],
        "p2p_commands": [],
        "metadata": {"f2p_stage": "test", "toolchain": "cpu"},
    }

    normalized = _normalize_toolchain_annotation(annotation, _instance())

    assert normalized["f2p_commands"] == [
        '"$(find build -type f -name Kokkos_CoreUnitTest_Serial1 '
        "-perm -111 -print -quit)\" --gtest_filter='*.numeric_traits_denorm_min'"
    ]


def test_death_test_uses_debug_build() -> None:
    annotation = {
        "configure_command": "cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "build_targets": ["CoreUnitTest_Serial1"],
        "metadata": {"f2p_stage": "test", "toolchain": "cpu"},
    }
    instance = replace(_instance(), test_patch='EXPECT_DEATH(foo(), "message");')

    normalized = _normalize_toolchain_annotation(annotation, instance)

    assert "-DCMAKE_BUILD_TYPE=Debug" in str(normalized["configure_command"])
    assert "RelWithDebInfo" not in str(normalized["configure_command"])


def test_tools_annotation_uses_full_build_for_dependencies() -> None:
    annotation = {
        "configure_command": "cmake -S . -B build",
        "build_command": "",
        "build_targets": ["test_one"],
        "metadata": {"toolchain": "cpu"},
    }
    instance = replace(_instance(), repo="kokkos/kokkos-tools")

    normalized = _normalize_toolchain_annotation(annotation, instance)

    assert normalized["build_targets"] == []
    assert normalized["build_command"] == "cmake --build build --parallel"


def test_pykokkos_annotation_allows_system_image_install() -> None:
    annotation = {
        "configure_command": "python -m pip install -e . --no-deps",
        "metadata": {"toolchain": "cpu"},
    }
    instance = replace(_instance(), repo="kokkos/pykokkos")

    normalized = _normalize_toolchain_annotation(annotation, instance)

    assert "--break-system-packages" in str(normalized["configure_command"])


def test_remote_spaces_annotation_uses_mpi_backend_options() -> None:
    annotation = {
        "configure_command": (
            "cmake -S . -B build -DBUILD_TESTING=ON -DKokkosRemoteSpaces_REMOTE_SPACES=MPI"
        ),
        "metadata": {"toolchain": "cpu"},
    }
    instance = replace(_instance(), repo="kokkos/kokkos-remote-spaces")

    normalized = _normalize_toolchain_annotation(annotation, instance)
    command = str(normalized["configure_command"])

    assert "KokkosRemoteSpaces_REMOTE_SPACES" not in command
    assert "-DBUILD_TESTING=ON" not in command
    assert "-DKRS_ENABLE_MPISPACE=ON" in command
    assert "-DKRS_ENABLE_TESTS=ON" in command


@pytest.mark.asyncio
async def test_failed_modal_validation_is_fed_back_for_repair() -> None:
    completer = _FakeCompleter([_annotation("BadTarget"), _annotation("GoodTarget")])
    sandboxes: list[_FakeSandbox] = []

    async def sandbox_factory(instance, timeout):
        sandbox = _FakeSandbox(instance.build_targets[0])
        sandboxes.append(sandbox)
        return sandbox

    result = await annotate_and_validate_instance(
        _instance(),
        completer=completer,
        max_attempts=2,
        flaky_repetitions=1,
        sandbox_factory=sandbox_factory,
    )

    assert result.passed
    assert result.validated_instance is not None
    assert result.validated_instance.build_targets == ("Kokkos_GoodTarget",)
    assert len(result.attempts) == 2
    assert result.attempts[0].validation is not None
    assert "unknown target BadTarget" in completer.prompts[1]
    assert all(sandbox.cleaned for sandbox in sandboxes)


def test_historical_subpackage_target_uses_base_cmake_and_preserves_test_selection():
    instance = replace(_instance(), merged_at="2020-01-01T00:00:00Z")
    annotation = {
        "build_targets": ["UnitTest_Serial"],
        "build_command": "ninja UnitTest_Serial",
        "metadata": {"toolchain": "cpu", "f2p_stage": "test"},
        "f2p_commands": ["./wrong --gtest_filter=*.UnorderedMap_valid_empty"],
        "p2p_commands": ["./wrong --gtest_filter=*.UnorderedMap_deep_copy"],
    }
    result = _normalize_toolchain_annotation(
        annotation,
        instance,
        build_context="KOKKOS_SUBPACKAGE(Containers)\nKOKKOS_ADD_EXECUTABLE_AND_TEST(\nUnitTest_${Tag}\n)",
    )
    assert result["build_targets"] == ["KokkosContainers_UnitTest_Serial"]
    assert result["build_command"] == ""
    assert "KokkosContainers_UnitTest_Serial" in result["f2p_commands"][0]
    assert "*.UnorderedMap_valid_empty" in result["f2p_commands"][0]
    assert "*.UnorderedMap_deep_copy" in result["p2p_commands"][0]


@pytest.mark.asyncio
async def test_adapter_timeout_does_not_spend_annotation_retry(monkeypatch):
    from unittest.mock import AsyncMock

    from . import auto_annotate as module
    from .validate import CommandResult, ValidationReport

    monkeypatch.setattr(module, "_fetch_cmake_context", AsyncMock(return_value=""))
    completer = _FakeCompleter([_annotation("GoodTarget"), _annotation("GoodTarget")])
    report = ValidationReport(
        instance_id=_instance().instance_id,
        passed=False,
        results=[
            CommandResult(
                command="cmake --build build",
                exit_code=-1,
                expected_exit="zero",
                stdout="",
                stderr="Command timed out after 300 seconds",
                seconds=300,
            )
        ],
        error="unexpected exit -1",
    )
    result = await annotate_and_validate_instance(
        _instance(),
        completer=completer,
        max_attempts=2,
        validation_runner=AsyncMock(return_value=report),
    )
    assert not result.passed
    assert len(completer.prompts) == 1
    assert len(result.attempts) == 1


def test_historical_kernels_names_follow_public_cmake_without_changing_selectors():
    context = "62: SET(PACKAGE_NAME KokkosKernels)\n163: SET(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})"
    instance = replace(_instance(), repo="kokkos/kokkos-kernels")
    annotation = {
        "metadata": {"toolchain": "cpu"},
        "build_targets": ["sparse_serial"],
        "build_command": "cmake --build build --target sparse_serial",
        "f2p_commands": [
            "./build/unit_test/serial/sparse_serial --gtest_filter='*spgemm*:*SPGEMM*'"
        ],
        "p2p_commands": [],
    }
    result = _normalize_toolchain_annotation(annotation, instance, build_context=context)
    assert result["build_targets"] == ["KokkosKernels_sparse_serial"]
    assert result["build_command"].endswith("--target KokkosKernels_sparse_serial")
    assert "-name KokkosKernels_sparse_serial" in result["f2p_commands"][0]
    assert result["f2p_commands"][0].endswith(" --gtest_filter='*spgemm*:*SPGEMM*'")


def test_preflight_target_binds_build_and_test_commands() -> None:
    instance = replace(
        _instance(),
        metadata={
            "runtime_test_inventory": {"target": "KokkosCore_UnitTest_Serial1", "binary": "b", "test_names": ["serial.a"]}
        },
    )
    finder = '"$(find build -type f -name {} -perm -111 -print -quit)" --gtest_filter=serial.a'
    annotation = {
        "build_command": "",
        "build_targets": ["Kokkos_UnitTest_Serial"],
        "f2p_commands": [finder.format("Kokkos_UnitTest_Serial")],
        "p2p_commands": [finder.format("Kokkos_UnitTest_Serial")],
        "metadata": {"toolchain": "cpu"},
    }
    normalized = _normalize_toolchain_annotation(annotation, instance)
    assert normalized["build_targets"] == ["KokkosCore_UnitTest_Serial1"]
    assert normalized["f2p_commands"] == [finder.format("KokkosCore_UnitTest_Serial1")]
    assert normalized["p2p_commands"] == [finder.format("KokkosCore_UnitTest_Serial1")]


def test_several_preflight_targets_bind_build_and_misnamed_commands() -> None:
    discovered = {"KokkosCore_UnitTest_Serial1": {"binary": "a", "test_names": ["serial.a"]},
                  "KokkosAlgorithms_UnitTest": {"binary": "b", "test_names": ["serial.b"]}}
    instance = replace(_instance(), metadata={"runtime_test_inventory": {"targets": discovered}})
    finder = '"$(find build -type f -name {} -perm -111 -print -quit)" --gtest_filter=serial.a'
    annotation = {
        "build_command": "",
        "build_targets": ["Kokkos_CoreUnitTest_Serial1", "KokkosAlgorithms_UnitTest"],
        "f2p_commands": [finder.format("Kokkos_CoreUnitTest_Serial1")],
        "p2p_commands": [finder.format("KokkosAlgorithms_UnitTest")],
        "metadata": {"toolchain": "cpu"},
    }
    normalized = _normalize_toolchain_annotation(annotation, instance)
    assert normalized["build_targets"] == ["KokkosAlgorithms_UnitTest", "KokkosCore_UnitTest_Serial1"]
    assert normalized["f2p_commands"] == [finder.format("KokkosCore_UnitTest_Serial1")]
    assert normalized["p2p_commands"] == [finder.format("KokkosAlgorithms_UnitTest")]
