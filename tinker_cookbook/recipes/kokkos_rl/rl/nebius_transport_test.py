import socket
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError

from tinker_cookbook.recipes.kokkos_rl.rl import nebius_transport as network


@pytest.mark.asyncio
async def test_lost_response_is_never_automatically_resampled(monkeypatch):
    requests = []

    def fail_after_send(request):
        requests.append(request)
        raise httpx.ReadError("Lost connection after request was sent", request=request)

    transport_builder = Mock(return_value=httpx.MockTransport(fail_after_send))
    monkeypatch.setattr(network.httpx, "AsyncHTTPTransport", transport_builder)
    policy = network.NetworkPolicy()
    async with network.create_nebius_client(
        api_key="test", base_url="https://example.test/v1", policy=policy
    ) as client:
        with pytest.raises(APIConnectionError):
            await client.chat.completions.create(
                model="test", messages=[{"role": "user", "content": "problem"}]
            )
    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["read"] is None
    assert requests[0].extensions["timeout"]["connect"] == 30.0
    transport_builder.assert_called_once_with(
        retries=0, socket_options=network.keepalive_socket_options(policy)
    )
    assert policy.identity()["unreceived_response_usage"].startswith("unknown")


@pytest.mark.parametrize("idle_name", ["TCP_KEEPIDLE", "TCP_KEEPALIVE"])
def test_linux_and_macos_keepalive_options(monkeypatch, idle_name):
    monkeypatch.delattr(socket, "TCP_KEEPIDLE", raising=False)
    monkeypatch.delattr(socket, "TCP_KEEPALIVE", raising=False)
    monkeypatch.setattr(socket, idle_name, 123, raising=False)
    policy = network.NetworkPolicy()
    options = network.keepalive_socket_options(policy)
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
    assert (socket.IPPROTO_TCP, 123, 60) in options
    assert (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10) in options
    assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3) in options


def test_unsupported_platform_fails_before_any_request(monkeypatch):
    monkeypatch.delattr(socket, "TCP_KEEPINTVL")
    with pytest.raises(RuntimeError, match="cannot enforce"):
        network.keepalive_socket_options(network.NetworkPolicy())


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_network_policy_rejected(seconds):
    with pytest.raises(ValueError, match="finite and positive"):
        network.NetworkPolicy(connect_timeout_seconds=seconds)
