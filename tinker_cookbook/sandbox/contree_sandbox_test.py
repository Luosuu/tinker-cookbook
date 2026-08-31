import asyncio
from types import SimpleNamespace

import contree_sdk._internals.models.instance as instance_module
import contree_sdk.sdk.objects.image_like._base as image_like_module

from tinker_cookbook.sandbox.contree_sandbox import (
    _ORIGINAL_INSTANCE_SPAWN_REQUEST,
    _network_aware_spawn_request,
    _spawn_without_network,
    _SpawnRequestWithNetworking,
)


def _request():
    return image_like_module.InstanceSpawnRequest(
        command="true",
        image="image",
        hostname="localhost",
        cwd="",
        stdin=None,
        truncate_output_at=1024,
        files={},
    )


def test_spawn_network_policy_is_isolated_between_concurrent_tasks(monkeypatch) -> None:
    monkeypatch.setattr(instance_module, "InstanceSpawnRequest", _ORIGINAL_INSTANCE_SPAWN_REQUEST)
    monkeypatch.setattr(image_like_module, "InstanceSpawnRequest", _ORIGINAL_INSTANCE_SPAWN_REQUEST)
    operation = object()
    client = SimpleNamespace(_operations={_ORIGINAL_INSTANCE_SPAWN_REQUEST: operation})

    async def exercise():
        blocked_entered = asyncio.Event()
        allowed_finished = asyncio.Event()

        async def blocked_request():
            with _spawn_without_network(client):
                blocked_entered.set()
                await allowed_finished.wait()
                return _request()

        async def allowed_request():
            await blocked_entered.wait()
            request = _request()
            allowed_finished.set()
            return request

        return await asyncio.gather(blocked_request(), allowed_request())

    blocked, allowed = asyncio.run(exercise())

    assert isinstance(blocked, _SpawnRequestWithNetworking)
    assert blocked.networking.enabled is False
    assert type(allowed) is _ORIGINAL_INSTANCE_SPAWN_REQUEST
    assert client._operations[_SpawnRequestWithNetworking] is operation
    assert instance_module.InstanceSpawnRequest is _network_aware_spawn_request
    assert image_like_module.InstanceSpawnRequest is _network_aware_spawn_request
