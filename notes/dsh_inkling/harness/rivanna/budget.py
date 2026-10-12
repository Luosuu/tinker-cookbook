"""Episode budget shared by every arm of a Rivanna block.

Budgets come from the named profiles in ``tinker_cookbook.recipes.kokkos_rl.budget``; the
default is the frozen ``v2.4`` budget. A block may start from another profile and override
single keys; designs written before a key existed get the ``v2.4`` value, which is what was
hard-coded then.
"""

from __future__ import annotations

from tinker_cookbook.recipes.kokkos_rl.budget import V24, KokkosBudget, budget


def from_profile(profile: KokkosBudget) -> dict[str, int | None]:
    return {
        "profile": profile.name,
        "max_model_requests": profile.max_turns,
        "max_tool_calls": profile.max_tool_calls,
        "max_sampled_tokens": profile.max_sampled_tokens,
        "max_input_tokens": profile.max_input_tokens,
        "max_context_tokens": profile.max_context_tokens,
        "wall_seconds": profile.wall_seconds,
        "command_timeout": profile.command_seconds,
        "grader_timeout": profile.grader_seconds,
        "sandbox_timeout": profile.sandbox_seconds,
        "build_check_max_runs": profile.build_check_max_runs,
        "build_check_max_seconds": profile.build_check_max_seconds,
        "native_max_infra_retries": 4,
    }


EPISODE_BUDGET = from_profile(V24)

# The DSH proxy and bridge enforce only the request, tool-call and time limits.
DSH_FIXED = ("max_sampled_tokens", "max_input_tokens", "max_context_tokens")


def resolve_budget(overrides: list[str] | dict | None = None, profile: str = "v2.4") -> dict:
    items = (
        overrides.items()
        if isinstance(overrides, dict)
        else (tuple(item.split("=", 1)) for item in (overrides or []))
    )
    resolved = from_profile(budget(profile))
    for key, value in items:
        if key not in resolved:
            raise ValueError(f"unknown budget key {key!r}")
        if key == "profile":
            resolved[key] = str(value)
        else:
            resolved[key] = None if value in (None, "none") else int(value)
    if resolved["sandbox_timeout"] < resolved["wall_seconds"] + resolved["grader_timeout"]:
        raise ValueError("sandbox_timeout must cover wall_seconds plus grader_timeout")
    return resolved
