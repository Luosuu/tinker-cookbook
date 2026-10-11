"""Decide when a Claude Code episode needs a bounded, neutral continuation."""

from __future__ import annotations


CONTINUE_PROMPT = (
    "Continue the original task in the same workspace. Your previous response ended "
    "after thinking without a tool action or final answer. Use the available MCP shell "
    "if you need to inspect or edit files, then complete the task."
)


def continuation_reason(
    quality: dict,
    *,
    policy: str,
    completed_continuations: int,
    max_continuations: int,
    model_turns: int,
    max_model_turns: int,
    tool_calls: int,
    max_tool_calls: int,
    remaining_seconds: float,
) -> str:
    """Return 'resume' or an auditable reason for stopping the episode."""
    if policy == "none":
        return "disabled"
    if policy != "thinking_only_resume":
        raise ValueError(f"Unknown continuation policy: {policy}")
    if quality.get("kind") != "thinking_only":
        return "other_completion"
    # Claude Code stream-json omits this field even when the upstream API's
    # stop_reason is end_turn; blank successful result is the available signal.
    if quality.get("assistant_stop_reason") not in {None, "end_turn"}:
        return "other_stop_reason"
    if quality.get("cli_result_subtype") != "success" or not quality.get("cli_result_blank"):
        return "cli_not_blank_success"
    if completed_continuations >= max_continuations:
        return "continuation_limit"
    if model_turns >= max_model_turns:
        return "model_turn_budget"
    if tool_calls >= max_tool_calls:
        return "tool_call_budget"
    if remaining_seconds <= 0:
        return "wall_time_budget"
    return "resume"


def sum_usage(results: list[dict]) -> dict:
    """Sum each CLI invocation's reported token counters, retaining raw records separately."""
    totals: dict[str, int] = {}
    for result in results:
        for key, value in (result.get("usage") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return totals
