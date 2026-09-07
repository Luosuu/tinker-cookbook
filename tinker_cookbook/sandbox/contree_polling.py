"""Bounded retries for read-only ConTree operation status requests.

The SDK cancels an operation if polling exits with a transport error. Retrying
that GET before it escapes preserves the operation ID and never resubmits work.
Callers must opt in and record the policy alongside their experiment identity.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from uuid import UUID

from contree_sdk import Contree
from contree_sdk._internals.client.client import ContreeClient
from contree_sdk.config import ContreeConfig
from contree_sdk.sdk.exceptions.api import ApiTimeoutError, ContreeTransportError
from httpx import Request, Response, TransportError

logger = logging.getLogger(__name__)
StatusResult = TypeVar("StatusResult")
_OPERATION_PATH = re.compile(r"/v1/operations/([0-9a-fA-F-]{36})$")


@dataclass(frozen=True)
class OperationPollPolicy:
    version: str = "readonly_status_retry_v1"
    max_attempts: int = 3
    retry_delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.version != "readonly_status_retry_v1":
            raise ValueError("Unknown operation polling policy")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("Operation status allows at most three GET attempts")
        if not math.isfinite(self.retry_delay_seconds) or not 0 <= self.retry_delay_seconds <= 5:
            raise ValueError("Operation status retry delay must be between zero and five seconds")


async def poll_operation_status(
    read_status: Callable[[str | UUID], Awaitable[StatusResult]],
    operation_id: str | UUID,
    policy: OperationPollPolicy,
) -> StatusResult:
    """Retry only transport failures of a GET for the same existing operation."""
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await read_status(operation_id)
        except (ApiTimeoutError, ContreeTransportError, TransportError) as error:
            if attempt == policy.max_attempts:
                raise
            # Never log exception text: it can contain request headers or payload.
            logger.warning(
                "ConTree status GET retry operation=%s attempt=%d/%d error_type=%s",
                operation_id,
                attempt,
                policy.max_attempts,
                type(error).__name__,
            )
            await asyncio.sleep(policy.retry_delay_seconds)
    raise AssertionError("Validated operation polling policy had no attempts")


class _PollingApiClient(ContreeClient):
    def __init__(self, config: ContreeConfig, policy: OperationPollPolicy) -> None:
        super().__init__(auth=config.auth, transport_timeout=config.transport_timeout)
        self.poll_policy = policy

    async def _send_request(self, request: Request) -> Response:
        match = _OPERATION_PATH.search(request.url.path)
        if request.method != "GET" or match is None:
            return await super()._send_request(request)
        operation_id = UUID(match[1])
        send_once = super()._send_request

        async def read_existing_status(existing_id: str | UUID) -> Response:
            if existing_id != operation_id:
                raise ValueError("Operation identity changed during status polling")
            return await send_once(request)

        return await poll_operation_status(read_existing_status, operation_id, self.poll_policy)


def create_polling_client(config: ContreeConfig, policy: OperationPollPolicy) -> Contree:
    """Create an opt-in SDK client without changing spawn, cancellation, or timeouts.

    Only the status GET endpoint is overridden. The supplied command/operation
    execution limits and HTTP timeout are preserved; this adds no spawn retry.
    """

    class PollingContree(Contree):
        @staticmethod
        def _create_api_client(config: ContreeConfig) -> ContreeClient:
            return _PollingApiClient(config, policy)

    return PollingContree(config=config)
