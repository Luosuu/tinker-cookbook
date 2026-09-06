"""Resume a known read-only chat prefix without drawing its model response again.

Use a separate client and result directory for each recovery. The caller must
verify the original task/config identity and explicitly approve every replayed
command. Only a single complete, untruncated tool turn is supported. The first
response is a recorded response, not a new provider generation. Compare the
new tool observations before allowing any live continuation request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import httpx
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

if TYPE_CHECKING:
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import OpenAITaskResult


@dataclass(frozen=True)
class ReadOnlyPrefix:
    response_id: str
    assistant_message: dict[str, object]
    raw_usage: dict[str, object]
    tool_messages: list[dict[str, object]]
    source_sha256: str

    @classmethod
    def from_transcript(
        cls, transcript: str, *, approved_commands: tuple[str, ...]
    ) -> ReadOnlyPrefix:
        records = json.loads(transcript)
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("Recovery supports exactly one complete tool turn")
        turn = records[0]
        if turn.get("turn") != 1 or turn.get("finish_reason") != "tool_calls":
            raise ValueError("Prefix must be the first completed tool-calling turn")
        assistant = ChatCompletionMessage.model_validate(turn["assistant_message"])
        calls = assistant.tool_calls or []
        if len(calls) != len(approved_commands) or not calls:
            raise ValueError("Every replayed command needs explicit approval")
        for call, approved in zip(calls, approved_commands, strict=True):
            if call.type != "function" or call.function.name != "bash":
                raise ValueError("Recovery supports only reviewed bash calls")
            arguments = json.loads(call.function.arguments)
            if arguments != {"command": approved}:
                raise ValueError("Recorded command differs from the approved read-only command")
        outputs = turn.get("tool_outputs", [])
        if len(outputs) != len(calls):
            raise ValueError("Every recorded tool call must have a complete observation")
        tool_messages: list[dict[str, object]] = []
        for call, output in zip(calls, outputs, strict=True):
            if (
                output.get("call_id") != call.id
                or output.get("model_output_truncated") is not False
            ):
                raise ValueError("Recovery requires ordered, untruncated tool observations")
            tool_messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output["output"]}
            )
        usage = CompletionUsage.model_validate(turn["raw_usage"])
        return cls(
            response_id=turn["response_id"],
            assistant_message=assistant.model_dump(exclude_none=True),
            raw_usage=usage.model_dump(),
            tool_messages=tool_messages,
            source_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
        )


class PrefixReplayTransport(httpx.AsyncBaseTransport):
    """Replay one original response, then require identical observed tool state."""

    def __init__(
        self,
        *,
        prefix: ReadOnlyPrefix,
        expected_first_request: dict[str, object],
        transport: httpx.AsyncBaseTransport,
    ) -> None:
        self.prefix = prefix
        self.expected_first_request = expected_first_request
        self.transport = transport
        self.replayed = False
        self.observations_verified = False
        self.live_requests = 0

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "source_transcript_sha256": self.prefix.source_sha256,
            "original_response_id": self.prefix.response_id,
            "cached_prefix_replayed": self.replayed,
            "replayed_tool_observations_verified": self.observations_verified,
            "live_request_attempts": self.live_requests,
            "original_prefix_usage": self.prefix.raw_usage,
            "accounting": "full recovered result includes original prefix once; subtract prefix usage before summing original and recovery phases",
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            raise ValueError("Recovery transport only accepts Chat Completions requests")
        payload = json.loads(request.content)
        if not self.replayed:
            if payload != self.expected_first_request:
                raise ValueError(
                    "Recovery request differs from the approved original configuration"
                )
            self.replayed = True
            return httpx.Response(
                200,
                json={
                    "id": self.prefix.response_id,
                    "object": "chat.completion",
                    # Original server creation time was not retained. This
                    # local replay is explicitly identified in provenance.
                    "created": 0,
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": self.prefix.assistant_message,
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": self.prefix.raw_usage,
                },
                headers={"x-kokkos-recorded-prefix": "true"},
            )
        if not self.observations_verified:
            initial = cast(list[dict[str, object]], self.expected_first_request["messages"])
            expected_messages = [
                *initial,
                self.prefix.assistant_message,
                *self.prefix.tool_messages,
            ]
            if payload["messages"] != expected_messages:
                raise ValueError("Replayed sandbox observations differ; no live continuation sent")
            if payload["model"] != self.expected_first_request["model"]:
                raise ValueError("Recovery model changed")
            self.observations_verified = True
        self.live_requests += 1
        return await self.transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self.transport.aclose()


def validate_recovery_source(original: OpenAITaskResult, prefix: ReadOnlyPrefix | None) -> None:
    """Only the observed missing-model rejection permits this bounded recovery."""
    if not original.error or not original.error.startswith("NotFoundError: Error code: 404"):
        raise ValueError("Recovery requires the recorded provider missing-model rejection")
    if prefix is None:
        if any(
            (
                original.turns_used,
                original.tool_calls,
                original.input_tokens,
                original.output_tokens,
            )
        ):
            raise ValueError("A fresh recovery may not discard any generated trajectory")
        return
    if (
        original.turns_used != 1
        or original.tool_calls != len(prefix.tool_messages)
        or original.input_tokens != prefix.raw_usage["prompt_tokens"]
        or original.output_tokens != prefix.raw_usage["completion_tokens"]
    ):
        raise ValueError("Recorded prefix does not account for the full original trajectory")


def recovery_accounting(
    original: OpenAITaskResult,
    recovered: OpenAITaskResult,
    *,
    prefix_replayed: bool,
) -> dict[str, object]:
    """Separate logical episode totals from tokens newly sent for inference."""
    from dataclasses import asdict

    source, logical = asdict(original), asdict(recovered)
    fields = ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens")
    incremental: dict[str, object] = {}
    for field in fields:
        value = logical[field] - (source[field] if prefix_replayed else 0)
        if value < 0:
            raise ValueError("Recovered usage is smaller than the replayed prefix")
        incremental[field] = value
    full_cost = recovered.estimated_cost_usd
    reused_cost = original.estimated_cost_usd if prefix_replayed else 0.0
    incremental["estimated_cost_usd"] = (
        round(full_cost - reused_cost, 6)
        if full_cost is not None and reused_cost is not None
        else None
    )
    if (
        isinstance(incremental["estimated_cost_usd"], float)
        and incremental["estimated_cost_usd"] < 0
    ):
        raise ValueError("Recovered cost is smaller than the replayed prefix")
    return {
        "prefix_replayed": prefix_replayed,
        "logical_trajectory_usage": {
            field: logical[field] for field in (*fields, "estimated_cost_usd")
        },
        "new_billable_usage": incremental,
        "original_attempt_usage": {
            field: source[field] for field in (*fields, "estimated_cost_usd")
        },
        "accounting_rule": "For cumulative spend, add original_attempt_usage and new_billable_usage; do not add both logical trajectory totals.",
    }
