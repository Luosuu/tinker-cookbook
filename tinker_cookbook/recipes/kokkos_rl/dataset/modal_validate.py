"""Validate annotated Kokkos candidates in fresh Modal sandboxes."""

from __future__ import annotations

import os
import re
import shlex
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext

from tinker_cookbook.recipes.kokkos_rl.dataset.ecosystem import get_repository_profile
from tinker_cookbook.recipes.kokkos_rl.dataset.models import KokkosInstance
from tinker_cookbook.recipes.kokkos_rl.dataset.validate import CommandResult, ValidationReport
from tinker_cookbook.sandbox import SandboxInterface

ERA_IMAGES = {
    "cpp14": "ubuntu:20.04",
    "cpp17": "ubuntu:22.04",
    "cpp20": "ubuntu:24.04",
}
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
HIP_IMAGE = "rocm/dev-ubuntu-22.04:6.4.3-complete"
SYCL_IMAGE = "intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04"
KOKKOS_DEPENDENCY_REF = "5.2.0"

ModalValidationSandboxFactory = Callable[[KokkosInstance, int], Awaitable[SandboxInterface]]

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _build_command(instance: KokkosInstance) -> str:
    if instance.build_command:
        return instance.build_command
    targets = " ".join(shlex.quote(target) for target in instance.build_targets)
    return f"cmake --build build --target {targets} --parallel"


def _cmake_test_source_diagnostic(instance: KokkosInstance) -> str:
    """Build a safe command that reveals how changed tests map to CMake targets."""

    stems = {
        os.path.splitext(os.path.basename(item.filename))[0]
        for item in instance.changed_files
        if any(part in item.filename.lower() for part in ("test", "unit_test"))
    }
    patterns = sorted({stem for value in stems for stem in (value, value.removeprefix("Test"))})
    patterns = [pattern for pattern in patterns if pattern]
    expression = "|".join(re.escape(pattern) for pattern in patterns) or "TESTNAMES|SOURCES"
    return (
        "grep -R -n -B 80 -A 20 -E "
        + shlex.quote(expression)
        + " --include=CMakeLists.txt . | head -n 300 || true"
    )


async def default_validation_sandbox_factory(
    instance: KokkosInstance, timeout: int
) -> SandboxInterface:
    """Create an image containing only the candidate's base checkout.

    Configuration and compilation intentionally happen after Sandbox creation so
    their logs are available to the annotation repair loop. Modal still caches the
    expensive clone layer by base commit.
    """

    import modal

    from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

    if instance.era not in ERA_IMAGES:
        raise ValueError(f"unsupported Kokkos compiler era: {instance.era!r}")
    if not _REPO_RE.fullmatch(instance.repo):
        raise ValueError(f"invalid GitHub repository name: {instance.repo!r}")
    if not _COMMIT_RE.fullmatch(instance.base_commit):
        raise ValueError(f"invalid base commit: {instance.base_commit!r}")

    clone_url = f"https://github.com/{instance.repo}.git"
    profile = get_repository_profile(instance.repo)
    accelerator = str(
        instance.metadata.get("toolchain", instance.metadata.get("accelerator", "cpu"))
    )
    requires_gpu = bool(instance.metadata.get("requires_gpu", False))
    if requires_gpu and accelerator in {"hip", "sycl"}:
        raise ValueError(
            f"Modal has no {accelerator} GPU worker for runtime validation; "
            "compile-only validation is supported"
        )
    gpu = (
        str(instance.metadata.get("modal_gpu", "L4"))
        if accelerator == "cuda" and requires_gpu
        else None
    )
    toolchain_images = {"cuda": CUDA_IMAGE, "hip": HIP_IMAGE, "sycl": SYCL_IMAGE}
    base_image = toolchain_images.get(accelerator, ERA_IMAGES[instance.era])
    clone_commands = [
        f"git clone {shlex.quote(clone_url)} /workspace/repo",
        "git -C /workspace/repo checkout " + shlex.quote(instance.base_commit),
        "git -C /workspace/repo submodule update --init --recursive",
    ]
    if instance.repo != "kokkos/kokkos" and profile.build_system == "cmake":
        dependency_configure = (
            "cmake -S /workspace/kokkos -B /workspace/kokkos-build -G Ninja "
            "-DCMAKE_INSTALL_PREFIX=/opt/kokkos -DKokkos_ENABLE_SERIAL=ON "
            "-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF "
            "-DCMAKE_BUILD_TYPE=Release"
        )
        if accelerator == "cuda":
            dependency_configure += (
                " -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_ADA89=ON "
                "-DCMAKE_CXX_COMPILER=/workspace/kokkos/bin/nvcc_wrapper"
            )
        elif accelerator == "hip":
            dependency_configure += (
                " -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_CXX_COMPILER=hipcc"
            )
        elif accelerator == "sycl":
            dependency_configure += (
                " -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON -DCMAKE_CXX_COMPILER=icpx"
            )
        clone_commands.extend(
            [
                "git clone --branch "
                + shlex.quote(KOKKOS_DEPENDENCY_REF)
                + " --depth 1 https://github.com/kokkos/kokkos.git /workspace/kokkos",
                dependency_configure,
                "cmake --build /workspace/kokkos-build --parallel",
                "cmake --install /workspace/kokkos-build",
            ]
        )
    if profile.build_system == "python":
        clone_commands.extend(
            [
                "python -m pip install --break-system-packages "
                "numpy patchelf pybind11 pytest setuptools wheel",
                "cd /workspace/repo && PIP_BREAK_SYSTEM_PACKAGES=1 "
                "python install_base.py install -- "
                "-DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF "
                "-DENABLE_VIEW_RANKS=4 -DENABLE_CUDA=OFF "
                "-DENABLE_THREADS=OFF -DENABLE_OPENMP=ON",
            ]
        )

    image = (
        modal.Image.from_registry(base_image)
        .apt_install(
            "build-essential",
            "ca-certificates",
            "ccache",
            "cmake",
            "git",
            "libboost-all-dev",
            "libhdf5-dev",
            "libopenmpi-dev",
            "ninja-build",
            "openmpi-bin",
            "python-is-python3",
            "python3-pip",
        )
        .run_commands(*clone_commands)
        .workdir("/workspace/repo")
    )
    output_context = (
        modal.enable_output() if os.environ.get("KOKKOS_MODAL_VERBOSE") else nullcontext()
    )
    with output_context:
        return await ModalSandbox.create(
            app_name="tinker-cookbook-kokkos-annotation",
            timeout=timeout,
            image=image,
            gpu=gpu,
            cpu=8.0 if gpu else 4.0,
            memory=32768 if gpu else 16384,
        )


async def _run(
    sandbox: SandboxInterface,
    command: str,
    *,
    timeout: int,
    expected_exit: str = "zero",
) -> CommandResult:
    started = time.monotonic()
    result = await sandbox.run_command(
        command,
        workdir="/workspace/repo",
        timeout=timeout,
    )
    return CommandResult(
        command=command,
        expected_exit=expected_exit,
        exit_code=result.exit_code,
        seconds=time.monotonic() - started,
        stdout=result.stdout[-16000:],
        stderr=result.stderr[-16000:],
    )


def _record(report: ValidationReport, result: CommandResult) -> None:
    report.results.append(result)
    if not result.matched_expectation:
        raise RuntimeError(
            f"unexpected exit {result.exit_code} for {result.command!r}; "
            f"expected {result.expected_exit}"
        )


async def _run_and_record(
    report: ValidationReport,
    sandbox: SandboxInterface,
    command: str,
    *,
    timeout: int,
    expected_exit: str = "zero",
) -> None:
    _record(
        report,
        await _run(
            sandbox,
            command,
            timeout=timeout,
            expected_exit=expected_exit,
        ),
    )


async def _write_patch(
    report: ValidationReport,
    sandbox: SandboxInterface,
    path: str,
    content: str,
    *,
    timeout: int,
) -> None:
    started = time.monotonic()
    result = await sandbox.write_file(path, content, timeout=timeout)
    _record(
        report,
        CommandResult(
            command=f"write {path}",
            expected_exit="zero",
            exit_code=result.exit_code,
            seconds=time.monotonic() - started,
            stdout=result.stdout[-16000:],
            stderr=result.stderr[-16000:],
        ),
    )


async def validate_instance_in_modal(
    instance: KokkosInstance,
    *,
    sandbox_timeout: int = 3600,
    command_timeout: int = 1200,
    flaky_repetitions: int = 3,
    sandbox_factory: ModalValidationSandboxFactory = default_validation_sandbox_factory,
) -> ValidationReport:
    """Run baseline, test-only failure, and gold-fix success in a Modal sandbox."""

    if not instance.is_validation_ready:
        raise ValueError(
            "instance needs build_targets, an F2P stage, test_patch, and code_patch "
            "before validation"
        )

    report = ValidationReport(instance_id=instance.instance_id)
    sandbox: SandboxInterface | None = None
    build_command = _build_command(instance)
    failure_stage = str(instance.metadata.get("f2p_stage", "test"))

    try:
        sandbox = await sandbox_factory(instance, sandbox_timeout)
        await _run_and_record(report, sandbox, instance.configure_command, timeout=command_timeout)
        baseline_build = await _run(sandbox, build_command, timeout=command_timeout)
        report.results.append(baseline_build)
        if not baseline_build.matched_expectation:
            # Give the annotation repair model concrete target-discovery evidence.
            report.results.append(
                await _run(
                    sandbox,
                    "cmake --build build --target help",
                    timeout=command_timeout,
                )
            )
            raise RuntimeError(
                f"unexpected exit {baseline_build.exit_code} for {build_command!r}; expected zero"
            )
        for command in instance.p2p_commands:
            await _run_and_record(report, sandbox, command, timeout=command_timeout)

        await _write_patch(
            report,
            sandbox,
            "/tmp/kokkos-test.patch",
            instance.test_patch,
            timeout=command_timeout,
        )
        await _run_and_record(
            report,
            sandbox,
            "git apply --whitespace=nowarn /tmp/kokkos-test.patch",
            timeout=command_timeout,
        )
        if failure_stage == "build":
            test_only_build = await _run(
                sandbox, build_command, timeout=command_timeout, expected_exit="nonzero"
            )
            report.results.append(test_only_build)
            if not test_only_build.matched_expectation:
                report.results.append(
                    await _run(
                        sandbox,
                        _cmake_test_source_diagnostic(instance),
                        timeout=command_timeout,
                    )
                )
                raise RuntimeError(
                    f"unexpected exit {test_only_build.exit_code} for {build_command!r}; "
                    "expected nonzero"
                )
        elif failure_stage == "test":
            await _run_and_record(report, sandbox, build_command, timeout=command_timeout)
            for _ in range(flaky_repetitions):
                for command in instance.f2p_commands:
                    await _run_and_record(
                        report,
                        sandbox,
                        command,
                        timeout=command_timeout,
                        expected_exit="nonzero",
                    )
        else:
            raise ValueError(f"unsupported metadata.f2p_stage: {failure_stage!r}")

        await _write_patch(
            report,
            sandbox,
            "/tmp/kokkos-code.patch",
            instance.code_patch,
            timeout=command_timeout,
        )
        await _run_and_record(
            report,
            sandbox,
            "git apply --whitespace=nowarn /tmp/kokkos-code.patch",
            timeout=command_timeout,
        )
        await _run_and_record(report, sandbox, build_command, timeout=command_timeout)
        for command in (*instance.f2p_commands, *instance.p2p_commands):
            await _run_and_record(report, sandbox, command, timeout=command_timeout)
        report.passed = True
    except Exception as error:
        report.error = f"{type(error).__name__}: {error}"
    finally:
        if sandbox is not None:
            try:
                await sandbox.cleanup()
            except Exception as error:
                if not report.error:
                    report.error = f"sandbox cleanup failed: {error}"
                    report.passed = False
    return report
