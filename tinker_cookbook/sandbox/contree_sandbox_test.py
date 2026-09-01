import asyncio
from datetime import timedelta
from types import SimpleNamespace

import contree_sdk._internals.models.instance as instance_module
import contree_sdk.sdk.objects.image_like._base as image_like_module

from tinker_cookbook.sandbox.contree_sandbox import (
    _ORIGINAL_INSTANCE_SPAWN_REQUEST,
    ContreeSandbox,
    _network_aware_spawn_request,
    _spawn_without_network,
    _SpawnRequestWithNetworking,
)
from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError


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


class _Session:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.uuid = "00000000-0000-0000-0000-000000000001"
        self.tag = None
        self._error = error

    async def run(self, **kwargs):
        del kwargs
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            stdout="ok\n", stderr="", exit_code=0, elapsed=timedelta(seconds=0.1)
        )


class _Image:
    def __init__(self, session: _Session) -> None:
        self._session = session

    def session(self) -> _Session:
        return self._session


def test_failed_operation_restores_last_successful_image() -> None:
    healthy_session = _Session()

    class Images:
        restored_image: str | None = None

        async def use(self, image: str) -> _Image:
            self.restored_image = image
            return _Image(healthy_session)

    images = Images()
    client = SimpleNamespace(images=images)
    sandbox = ContreeSandbox(
        client=client,
        image=_Image(_Session(error=RuntimeError("operation timed out"))),
        timeout=600,
    )

    async def exercise():
        failed = await sandbox.run_command("slow build")
        recovered = await sandbox.run_command("echo alive")
        return failed, recovered

    failed, recovered = asyncio.run(exercise())

    assert failed.exit_code == -1
    assert failed.metrics == {"sandbox_recovered": True}
    assert images.restored_image == "00000000-0000-0000-0000-000000000001"
    assert recovered.exit_code == 0
    assert recovered.stdout == "ok\n"


def test_failed_recovery_terminates_sandbox() -> None:
    class Images:
        async def use(self, image: str) -> _Image:
            del image
            raise RuntimeError("restore failed")

    sandbox = ContreeSandbox(
        client=SimpleNamespace(images=Images()),
        image=_Image(_Session(error=RuntimeError("operation timed out"))),
        timeout=600,
    )

    async def exercise() -> None:
        await sandbox.run_command("slow build")

    try:
        asyncio.run(exercise())
    except SandboxTerminatedError as error:
        assert "operation timed out" in str(error)
        assert "restore failed" in str(error)
    else:
        raise AssertionError("expected a SandboxTerminatedError")
