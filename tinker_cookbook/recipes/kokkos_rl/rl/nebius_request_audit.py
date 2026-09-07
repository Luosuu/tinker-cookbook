"""Persist each generation boundary without repeating an uncertain request."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


def write_exclusive(path: Path, value: object) -> None:
    """A prior file, including a partial write, always prevents another attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class AuditedCompletions:
    def __init__(
        self,
        create: Callable[..., Awaitable[ChatCompletion]],
        directory: Path,
        identity_sha256: str,
    ) -> None:
        if directory.exists():
            raise ValueError("Prior generation audit exists; recovery needs explicit review")
        self._create = create
        self.directory = directory
        self.identity_sha256 = identity_sha256
        self._next = 1

    async def create(self, **kwargs: object) -> ChatCompletion:
        if kwargs.get("stream") not in (None, False):
            raise ValueError("This audited evaluation requires complete non-stream responses")
        number = self._next
        self._next += 1
        folder = self.directory / f"{number:03d}"
        request = {
            "phase_identity_sha256": self.identity_sha256,
            "request_number": number,
            "started_at": datetime.now(UTC).isoformat(),
            "arguments": kwargs,
            "automatic_retries": 0,
        }
        write_exclusive(folder / "request.json", request)
        try:
            response = await self._create(**kwargs)
            # Preserve the provider response even when a later usage/protocol
            # check rejects it. Never infer that an absent usage field means zero.
            payload = response.model_dump(mode="json")
            write_exclusive(folder / "response.json", payload)
        except BaseException as error:
            write_exclusive(
                folder / "unreceived_or_unpersisted_response.json",
                {
                    "error_type": type(error).__name__,
                    "request_number": number,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "usage": "unknown",
                    "retry_permitted": False,
                },
            )
            raise
        write_exclusive(
            folder / "completion.json",
            {
                "request_number": number,
                "finished_at": datetime.now(UTC).isoformat(),
                "response_sha256": hashlib.sha256(
                    (folder / "response.json").read_bytes()
                ).hexdigest(),
                "usage_available": response.usage is not None,
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
                "received_response_count": 1,
            },
        )
        return response


def audited_client(client: AsyncOpenAI, directory: Path, identity_sha256: str) -> SimpleNamespace:
    """Wrap only the chat entrypoint used by the new, explicitly pinned phase."""
    if client.max_retries != 0:
        raise ValueError("Generation auditing requires SDK retries to be disabled")
    create = cast(Callable[..., Awaitable[ChatCompletion]], client.chat.completions.create)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=AuditedCompletions(create, directory, identity_sha256))
    )
