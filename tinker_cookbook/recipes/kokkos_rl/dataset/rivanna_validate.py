"""Rivanna-only construction backend using the existing Apptainer adapter.

Run inside a bounded Slurm allocation. KOKKOS_RIVANNA_BASE_SIF must be an
existing image with the OpenHands server and required compiler tools; CUDA tasks use a
CUDA development image of their era and run in a GPU allocation.
Every task receives a fresh writable container and a fresh upstream checkout.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from uuid import uuid4

from tinker_cookbook.sandbox.apptainer_sandbox import ApptainerSandbox

from .export_harbor import _clean_room_command
from .modal_validate import _COMMIT_RE, _REPO_RE
from .models import KokkosInstance


class BuildEvidenceSandbox:
    """Preserve bounded build output even when the workspace adapter times out."""

    def __init__(self, sandbox):
        self.sandbox = sandbox

    def __getattr__(self, name):
        return getattr(self.sandbox, name)

    async def run_command(self, command, workdir=None, timeout=60, max_output_bytes=None):
        if command.lstrip().startswith("cmake --build "):
            log = "/tmp/kokkos-build-" + uuid4().hex + ".log"
            script = "export NINJA_STATUS='[%f/%t %e sec] '; time " + command
            wrapped = (
                f"timeout --kill-after=5s {int(timeout)}s bash -c {shlex.quote(script)} > {log} 2>&1; "
                f"status=$?; tail -c 120000 {log}; "
                f'printf \'\\nKOKKOS_BUILD_EXIT=%s TIME_LIMIT_SECONDS={int(timeout)}\\n\' "$status"; exit "$status"'
            )
            return await self.sandbox.run_command(
                wrapped, workdir=workdir, timeout=timeout + 15, max_output_bytes=max_output_bytes
            )
        return await self.sandbox.run_command(
            command, workdir=workdir, timeout=timeout, max_output_bytes=max_output_bytes
        )


async def rivanna_validation_sandbox_factory(
    instance: KokkosInstance, timeout: int
) -> ApptainerSandbox:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME") not in {
        "rivanna",
        "shen",
    }:
        raise RuntimeError("Construction validation requires a Rivanna Slurm allocation")
    toolchain = instance.metadata.get("toolchain", "cpu")
    if instance.repo not in {"kokkos/kokkos", "kokkos/kokkos-kernels"} or not (
        toolchain == "cpu" or (toolchain == "cuda" and instance.repo == "kokkos/kokkos")
    ):
        raise ValueError(
            "This Rivanna construction profile supports CPU core/kernels and CUDA core"
        )
    gpu = toolchain == "cuda"
    if gpu and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("CUDA construction requires a Slurm GPU allocation")
    dependency = instance.metadata.get("kokkos_dependency_commit", "")
    if instance.repo == "kokkos/kokkos-kernels" and not _COMMIT_RE.fullmatch(dependency):
        raise ValueError("Historical kernels environment requires an immutable Kokkos dependency")
    if not _REPO_RE.fullmatch(instance.repo) or not _COMMIT_RE.fullmatch(instance.base_commit):
        raise ValueError("Invalid upstream identity")
    from .qualification import execution_policy

    parallelism = str(execution_policy(instance)["build_parallelism"])
    path = Path(os.environ["KOKKOS_RIVANNA_BASE_SIF"]).resolve(strict=True)
    sandbox = await ApptainerSandbox.create(
        path,
        timeout,
        default_workdir="/workspace/repo",
        enable_gpu=gpu,
        command_env={"CMAKE_BUILD_PARALLEL_LEVEL": parallelism, "OMP_NUM_THREADS": "2"},
    )
    try:
        probe = await sandbox.run_command(
            "command -v cmake && command -v ninja && command -v ccache && command -v g++"
            + (" && command -v nvcc && nvidia-smi -L" if gpu else ""),
            workdir="/",
            timeout=30,
        )
        if probe.exit_code:
            raise RuntimeError("Rivanna environment profile is missing required compiler tools")
        if instance.repo == "kokkos/kokkos-kernels":
            result = await sandbox.run_command(
                "git init /workspace/kokkos && cd /workspace/kokkos && "
                "git remote add origin https://github.com/kokkos/kokkos.git && "
                f"git fetch --depth=1 origin {shlex.quote(dependency)} && git checkout --detach FETCH_HEAD && "
                "cmake -S . -B /workspace/kokkos-build -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/kokkos "
                "-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF "
                "-DCMAKE_BUILD_TYPE=Release && cmake --build /workspace/kokkos-build --parallel 2 && "
                "cmake --install /workspace/kokkos-build",
                workdir="/",
                timeout=min(timeout, 600),
            )
            if result.exit_code:
                raise RuntimeError(
                    "Historical kernels dependency environment failed: "
                    + result.stdout[-1500:]
                    + result.stderr[-1500:]
                )
        result = await sandbox.run_command(
            "rm -rf /workspace/repo && mkdir -p /workspace/repo && cd /workspace/repo && "
            f"git init . && git remote add origin https://github.com/{instance.repo}.git && "
            f"git fetch --depth=1 origin {shlex.quote(instance.base_commit)} && "
            "git checkout --detach FETCH_HEAD && git submodule update --init --recursive && "
            + _clean_room_command(instance.base_commit),
            workdir="/",
            timeout=min(timeout, 300),
        )
        if result.exit_code:
            raise RuntimeError("Rivanna checkout failed: " + result.stderr[-2000:])
        return BuildEvidenceSandbox(sandbox)
    except BaseException:
        await sandbox.cleanup()
        raise
