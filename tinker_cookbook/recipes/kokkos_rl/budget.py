"""Named SWE-kokkos-bench episode budgets.

Every evaluator, the Rivanna block runner and the Pi agent read their limits from one of
these, so a budget is changed in one place and a result can name the budget it ran under.

``v2.4`` is the frozen budget of the published v2.4 leaderboard and of RFT self-training.
``long-250`` lets the request cap and the model context window be the only binding limits:
every other limit is either derived from those two or set well above what 250-request
episodes used on Rivanna (block-t01, 373 episodes; see notes/pi_harness/BUDGET.md). Episodes
stopped on their own by request 102 at the latest, so 250 does not bind either.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

CONTEXT_WINDOW = 262_144  # Inkling-Small:peft:262144


@dataclass(frozen=True)
class KokkosBudget:
    name: str
    max_turns: int
    max_tool_calls: int
    max_tokens: int  # per response
    max_sampled_tokens: int  # cumulative over the episode
    max_input_tokens: int  # cumulative over the episode
    max_context_tokens: int | None  # per request; None disables the check
    max_tool_output_chars: int
    wall_seconds: int
    command_seconds: int
    grader_seconds: int
    sandbox_seconds: int
    build_check_max_runs: int
    build_check_max_seconds: int

    def as_dict(self) -> dict:
        return asdict(self)


V24 = KokkosBudget(
    name="v2.4",
    max_turns=40,
    max_tool_calls=80,
    max_tokens=16_384,
    max_sampled_tokens=65_536,
    max_input_tokens=5_000_000,
    max_context_tokens=None,
    max_tool_output_chars=16_384,
    wall_seconds=3_600,
    command_seconds=900,
    grader_seconds=2_400,
    sandbox_seconds=7_200,
    build_check_max_runs=8,
    build_check_max_seconds=1_200,
)

_LONG_TURNS = 250
LONG_250 = KokkosBudget(
    name="long-250",
    max_turns=_LONG_TURNS,
    # One tool call per request was measured in 14,155 of 14,155 responses; keep the 2:1 ratio.
    max_tool_calls=2 * _LONG_TURNS,
    max_tokens=16_384,
    # Measured maximum 25,191 output tokens per episode.
    max_sampled_tokens=131_072,
    # The context guard bounds every prompt by the window, so this cannot bind first.
    max_input_tokens=_LONG_TURNS * CONTEXT_WINDOW,
    max_context_tokens=CONTEXT_WINDOW,
    max_tool_output_chars=16_384,
    # Measured maximum 1,787 s, with eight build checks.
    wall_seconds=7_200,
    command_seconds=900,
    grader_seconds=2_400,
    # A full episode plus a verifier build, with margin.
    sandbox_seconds=10_800,
    # Eight checks and 1,200 s were reached at the 99th percentile; triple both.
    build_check_max_runs=24,
    build_check_max_seconds=3_600,
)

BUDGETS: dict[str, KokkosBudget] = {b.name: b for b in (V24, LONG_250)}


def budget(name: str) -> KokkosBudget:
    if name not in BUDGETS:
        raise ValueError(f"unknown budget {name!r}; choose from {sorted(BUDGETS)}")
    return BUDGETS[name]
