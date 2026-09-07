"""Explicit network policy for future Nebius evaluation identities.

TCP keepalive detects an unreachable peer without imposing a generation-time
deadline. Disable both HTTP and SDK retries: a lost response can have unknown
billable usage and must not silently draw another answer. This does not repair
already-open sockets or authorize replay of an incomplete trajectory.
"""

from __future__ import annotations

import math
import socket
from dataclasses import asdict, dataclass

import httpx
from openai import AsyncOpenAI


@dataclass(frozen=True)
class NetworkPolicy:
    keepalive_idle_seconds: int = 60
    keepalive_interval_seconds: int = 10
    keepalive_probe_count: int = 3
    connect_timeout_seconds: float = 30.0
    write_timeout_seconds: float = 60.0
    pool_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for value in (
            self.keepalive_idle_seconds,
            self.keepalive_interval_seconds,
            self.keepalive_probe_count,
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("TCP keepalive values must be integers")
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Network policy values must be finite and positive")

    def identity(self) -> dict[str, object]:
        """Include this policy in a new phase identity before sending requests."""
        return {
            "policy": "tcp_keepalive_no_request_retries_v1",
            **asdict(self),
            "read_timeout_seconds": None,
            "http_transport_retries": 0,
            "sdk_retries": 0,
            "unreceived_response_usage": "unknown; requires explicit recovery review",
        }


def keepalive_socket_options(policy: NetworkPolicy) -> list[tuple[int, int, int]]:
    # Linux names this TCP_KEEPIDLE; macOS names it TCP_KEEPALIVE.
    idle_option = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    interval_option = getattr(socket, "TCP_KEEPINTVL", None)
    count_option = getattr(socket, "TCP_KEEPCNT", None)
    if idle_option is None or interval_option is None or count_option is None:
        raise RuntimeError("This platform cannot enforce the declared TCP keepalive policy")
    return [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, idle_option, policy.keepalive_idle_seconds),
        (socket.IPPROTO_TCP, interval_option, policy.keepalive_interval_seconds),
        (socket.IPPROTO_TCP, count_option, policy.keepalive_probe_count),
    ]


def create_nebius_client(
    *, api_key: str, base_url: str, policy: NetworkPolicy = NetworkPolicy()
) -> AsyncOpenAI:
    """Create a client for an explicitly recorded policy; close with async with."""
    timeout = httpx.Timeout(
        connect=policy.connect_timeout_seconds,
        read=None,
        write=policy.write_timeout_seconds,
        pool=policy.pool_timeout_seconds,
    )
    transport = httpx.AsyncHTTPTransport(retries=0, socket_options=keepalive_socket_options(policy))
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=timeout,
        http_client=httpx.AsyncClient(transport=transport, timeout=timeout),
    )
