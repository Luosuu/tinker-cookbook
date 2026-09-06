"""Explicit per-task sandbox resources for reproducible benchmark comparisons."""

from __future__ import annotations

import math
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import SandboxFactory
from tinker_cookbook.sandbox import SandboxInterface


async def _create_modal_sandbox(
    env_dir: Path, timeout: int, *, memory_mb: int, cpu: float, build_parallelism: int = 1
) -> SandboxInterface:
    import modal

    from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

    image = modal.Image.from_dockerfile(
        path=str(env_dir / "Dockerfile"), context_dir=str(env_dir)
    ).env({"CMAKE_BUILD_PARALLEL_LEVEL": str(build_parallelism)})
    return await ModalSandbox.create(
        image=image,
        timeout=timeout,
        memory=memory_mb,
        cpu=cpu,
        allow_network=False,
    )


def create_resource_sandbox_factory(
    primary_factory: SandboxFactory,
    modal_task_names: tuple[str, ...],
    *,
    memory_mb: int = 16384,
    cpu: float = 4.0,
    build_parallelism: int = 1,
) -> SandboxFactory:
    """Route named tasks before sampling, never based on the model's outcome.

    Callers must include this policy in their experiment identity and validate
    Oracle/NOP on the selected backend. Use the same factory for model rollouts
    and any fresh patch regrade. Non-selected tasks retain the primary backend.
    """
    if build_parallelism < 1:
        raise ValueError("Build parallelism must be positive")
    if memory_mb < 1 or not math.isfinite(cpu) or cpu <= 0:
        raise ValueError("Sandbox memory and CPU must be positive")
    if len(set(modal_task_names)) != len(modal_task_names):
        raise ValueError("Modal task names must be unique")
    if any(not name or Path(name).name != name for name in modal_task_names):
        raise ValueError("Modal task names must be plain directory names")
    selected = frozenset(modal_task_names)

    async def factory(env_dir: Path, timeout: int) -> SandboxInterface:
        if env_dir.parent.name in selected:
            return await _create_modal_sandbox(
                env_dir, timeout, memory_mb=memory_mb, cpu=cpu, build_parallelism=build_parallelism
            )
        return await primary_factory(env_dir, timeout)

    return factory
