"""Episode budget shared by every arm of a Rivanna block.

The defaults are the frozen SWE-kokkos-bench budget (40 model requests). A block may
override them for a budget study; designs written before a key existed get its default,
which is the value that was hard-coded then.
"""

from __future__ import annotations

EPISODE_BUDGET: dict[str, int | None] = {
    "max_model_requests": 40,
    "max_tool_calls": 80,
    "max_sampled_tokens": 65536,
    "max_input_tokens": 5_000_000,
    # None: no per-request context check (the frozen budget never approaches the window).
    "max_context_tokens": None,
    "wall_seconds": 3600,
    "command_timeout": 900,
    # The sandbox must outlive a full episode plus a verifier build.
    "grader_timeout": 2400,
    "sandbox_timeout": 7200,
    "native_max_infra_retries": 4,
}

# The DSH proxy and bridge enforce only the request, tool-call and time limits.
DSH_FIXED = ("max_sampled_tokens", "max_input_tokens", "max_context_tokens")


def resolve_budget(overrides: list[str] | dict | None = None) -> dict:
    items = (
        overrides.items()
        if isinstance(overrides, dict)
        else (tuple(item.split("=", 1)) for item in (overrides or []))
    )
    budget = dict(EPISODE_BUDGET)
    for key, value in items:
        if key not in EPISODE_BUDGET:
            raise ValueError(f"unknown budget key {key!r}")
        budget[key] = None if value in (None, "none") else int(value)
    if budget["sandbox_timeout"] < budget["wall_seconds"] + budget["grader_timeout"]:
        raise ValueError("sandbox_timeout must cover wall_seconds plus grader_timeout")
    return budget
