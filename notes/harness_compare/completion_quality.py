"""Classify the final Claude Code stream event without changing verifier scores."""

from __future__ import annotations

import json
from collections.abc import Iterable


def classify_stream(lines: Iterable[str]) -> dict[str, object]:
    final_message: dict | None = None
    cli_result: dict | None = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            final_message = event.get("message") or {}
        elif event.get("type") == "result":
            cli_result = event

    blocks = final_message.get("content", []) if final_message is not None else []
    types = [block.get("type") for block in blocks]
    text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    if final_message is None:
        kind = "missing_assistant"
    elif types and set(types) == {"thinking"}:
        kind = "thinking_only"
    elif text and "mcp__" in text and '"args"' in text:
        kind = "tool_call_as_text"
    elif text.strip():
        kind = "final_text"
    elif "tool_use" in types:
        kind = "pending_tool_use"
    else:
        kind = "empty_assistant"
    return {
        "kind": kind,
        "content_types": types,
        "assistant_stop_reason": final_message.get("stop_reason") if final_message else None,
        "cli_result_subtype": cli_result.get("subtype") if cli_result else None,
        "cli_result_blank": not bool((cli_result or {}).get("result", "").strip()) if cli_result else None,
    }
