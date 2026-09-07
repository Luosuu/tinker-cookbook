import asyncio
from dataclasses import asdict
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from contree_sdk._internals.client.client import ContreeClient
from contree_sdk._internals.models.instance import (
    InstanceOperationMetadata,
    InstanceOperationResult,
)
from contree_sdk._internals.models.operation import OperationKind, OperationModel
from contree_sdk.auth import JWTAuth
from contree_sdk.config import ContreeConfig
from contree_sdk.sdk.exceptions.api import (
    ApiStatusCodeError,
    ApiTimeoutError,
    ContreeTransportError,
)
from contree_sdk.utils.models.operation import OperationStatus
from contree_sdk.utils.models.stream import StreamDescription
from httpx import ReadTimeout, Request, Response

from tinker_cookbook.sandbox.contree_polling import (
    OperationPollPolicy,
    create_polling_client,
    poll_operation_status,
)


def completed_operation():
    metadata = InstanceOperationMetadata(
        args=[],
        command="bash /tests/test.sh",
        cwd="/workspace/repo",
        disposable=False,
        env={},
        files={},
        hostname="test",
        image="input-image",
        result=None,
        shell=True,
        stdin=StreamDescription(""),
        timeout=900,
        truncate_output_at=65536,
    )
    return OperationModel(
        kind=OperationKind.INSTANCE,
        status=OperationStatus.SUCCESS,
        duration=5,
        metadata=metadata,
        result=InstanceOperationResult(image="result-image", tag=None),
    )


@pytest.mark.asyncio
async def test_transport_retries_keep_same_id_and_return_original_result():
    result = completed_operation()
    read = AsyncMock(
        side_effect=[ApiTimeoutError(timeout_type="read"), ContreeTransportError(), result]
    )
    oid = UUID("11111111-1111-1111-1111-111111111111")
    assert (
        await poll_operation_status(read, oid, OperationPollPolicy(retry_delay_seconds=0)) is result
    )
    assert [call.args for call in read.await_args_list] == [(oid,), (oid,), (oid,)]


@pytest.mark.asyncio
async def test_transport_retries_are_finite_and_keep_original_error():
    error = ApiTimeoutError(timeout_type="read")
    read = AsyncMock(side_effect=error)
    with pytest.raises(ApiTimeoutError) as raised:
        await poll_operation_status(read, "same-id", OperationPollPolicy(retry_delay_seconds=0))
    assert raised.value is error
    assert read.await_count == 3


@pytest.mark.parametrize(
    "error", [ApiStatusCodeError(status=401), ValueError("bad payload"), asyncio.CancelledError()]
)
@pytest.mark.asyncio
async def test_nontransport_errors_and_cancellation_are_not_retried(error):
    read = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await poll_operation_status(read, "same-id", OperationPollPolicy(retry_delay_seconds=0))
    assert read.await_count == 1


@pytest.mark.asyncio
async def test_sdk_wait_survives_get_error_without_cancel_or_spawn(monkeypatch):
    result = completed_operation()
    response = Response(
        200,
        json=asdict(result),
        request=Request(
            "GET", "https://example.invalid/v1/operations/11111111-1111-1111-1111-111111111111"
        ),
    )
    read = AsyncMock(side_effect=[ReadTimeout("read timeout"), response])
    spawn = AsyncMock()
    cancel = AsyncMock()
    monkeypatch.setattr(ContreeClient, "_send_request", read)
    monkeypatch.setattr(ContreeClient, "spawn_instance", spawn)
    monkeypatch.setattr(ContreeClient, "cancel_operation", cancel)
    config = ContreeConfig(
        auth=JWTAuth(token="test-only-token", base_url="https://example.invalid"),
        transport_timeout=10,
        operation_timeout=900,
        operation_run_timeout=900,
    )
    policy = OperationPollPolicy(retry_delay_seconds=0)
    client = create_polling_client(config, policy)
    oid = UUID("11111111-1111-1111-1111-111111111111")
    metadata, image = await client._wait_operation(oid, InstanceOperationMetadata, timeout=900)
    assert metadata.timeout == 900 and image.image == "result-image"
    requests = [call.args[0] for call in read.await_args_list]
    assert requests[0] is requests[1]
    assert requests[0].method == "GET" and str(oid) in requests[0].url.path
    spawn.assert_not_awaited()
    cancel.assert_not_awaited()
    assert client.config.transport_timeout == 10
    assert client.config.operation_timeout == client.config.operation_run_timeout == 900
    assert asdict(policy)["max_attempts"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": 4},
        {"retry_delay_seconds": float("inf")},
        {"version": "unknown"},
    ],
)
def test_unbounded_or_unknown_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OperationPollPolicy(**kwargs)


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/v1/instances"),
        ("DELETE", "/v1/operations/11111111-1111-1111-1111-111111111111"),
        ("GET", "/v1/images"),
    ],
)
@pytest.mark.asyncio
async def test_other_endpoints_never_receive_transport_retries(monkeypatch, method, path):
    read = AsyncMock(side_effect=ReadTimeout("read timeout"))
    monkeypatch.setattr(ContreeClient, "_send_request", read)
    config = ContreeConfig(
        auth=JWTAuth(token="test-only-token", base_url="https://example.invalid")
    )
    client = create_polling_client(config, OperationPollPolicy(retry_delay_seconds=0))
    with pytest.raises(ReadTimeout):
        await client._api._send_request(Request(method, "https://example.invalid" + path))
    assert read.await_count == 1
