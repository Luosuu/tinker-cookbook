from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl import resource_sandbox


@pytest.mark.asyncio
async def test_static_resource_routing_preserves_primary_backend(monkeypatch):
    primary = AsyncMock()
    modal = AsyncMock()
    monkeypatch.setattr(resource_sandbox, "_create_modal_sandbox", modal)
    factory = resource_sandbox.create_resource_sandbox_factory(
        primary, ("large",), memory_mb=16384, cpu=4.0
    )
    normal_path = Path("/snapshot/normal/environment")
    large_path = Path("/snapshot/large/environment")
    assert await factory(normal_path, 3600) is primary.return_value
    assert await factory(large_path, 3600) is modal.return_value
    primary.assert_awaited_once_with(normal_path, 3600)
    modal.assert_awaited_once_with(large_path, 3600, memory_mb=16384, cpu=4.0)


@pytest.mark.parametrize("names", [("a", "a"), ("../a",), ("",)])
def test_invalid_task_policy_rejected(names):
    with pytest.raises(ValueError):
        resource_sandbox.create_resource_sandbox_factory(AsyncMock(), names)


@pytest.mark.parametrize("memory,cpu", [(0, 4), (16384, 0), (16384, float("nan"))])
def test_invalid_resources_rejected(memory, cpu):
    with pytest.raises(ValueError):
        resource_sandbox.create_resource_sandbox_factory(AsyncMock(), (), memory_mb=memory, cpu=cpu)


@pytest.mark.asyncio
async def test_modal_receives_memory_cpu_and_network_policy(monkeypatch):
    from unittest.mock import Mock

    image = Mock()
    builder = Mock(return_value=image)
    create = AsyncMock()
    monkeypatch.setattr("modal.Image.from_dockerfile", builder)
    monkeypatch.setattr("tinker_cookbook.sandbox.modal_sandbox.ModalSandbox.create", create)
    await resource_sandbox._create_modal_sandbox(
        Path("/snapshot/large/environment"), 3600, memory_mb=16384, cpu=4.0
    )
    image.env.assert_called_once_with({"CMAKE_BUILD_PARALLEL_LEVEL": "1"})
    create.assert_awaited_once_with(
        image=image.env.return_value, timeout=3600, memory=16384, cpu=4.0, allow_network=False
    )
