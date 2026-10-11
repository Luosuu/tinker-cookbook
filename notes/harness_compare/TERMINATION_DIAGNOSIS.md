# Claude Code / Inkling Small premature termination diagnosis

The full 100-task evaluation produced real, complete verifier results for all
400 candidates, but its Claude Code sessions did not end normally. Reclassifying
the saved `stream-json` events gives **379/400 thinking-only final messages**
and **21/400 final text messages that contain unexecuted MCP-call syntax**.
Every CLI result reported `subtype=success`; the 379 thinking-only results had
blank result text. There was no ordinary final answer in these 400 attempts.
These are completion-protocol diagnostics, not a retrospective change to the
verifier scores.

| Final event and candidate | Attempts | Passing |
| --- | ---: | ---: |
| Thinking only, no production edit | 314 | 0 |
| Tool-call syntax as text, no production edit | 19 | 0 |
| Thinking only, production edit | 65 | 45 |
| Tool-call syntax as text, production edit | 2 | 2 |

This explains the low interaction counts: median seven model turns and six
executed sandbox calls, with a maximum of 33 model turns against the frozen
40-turn cap. The 333 attempts with no production edit all failed grading.
Candidate capture used the baseline Git diff plus untracked non-build files and
reported complete artifacts; the evidence does not point to missing patch
capture, broken MCP access, budget exhaustion, or grader infrastructure errors.

To locate the termination boundary, job `20522440` ran the same Claude Code
`2.1.50` CLI and model against a fake MCP shell while a local reverse proxy
recorded only Anthropic request/response shapes. The first five upstream
responses carried `thinking + tool_use` with `stop_reason=tool_use`. The sixth
returned only `thinking` with `stop_reason=end_turn`. Claude Code then exited
zero with `subtype=success`, six turns, and blank result text. Thus the CLI
followed the endpoint's terminal response; this was not a dropped MCP call in
the CLI or bridge. The wire log is
[`wire-20522440.log`](../artifacts/claude-code-pass4-20260924/diagnostics/wire-20522440.log).
The probe uses canned shell outputs and does not produce a benchmark score.

Claude Code's request sent `thinking.budget_tokens=31999` and no
`output_config.effort`. [Tinker's Anthropic API documentation](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/)
says the Anthropic thinking-token budget is ignored and effort is controlled by
`output_config.effort`; [Inkling's effort guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/)
lists high/0.9 as the default. Missing an explicit effort is therefore **not
established as the cause**. In job `20522451`, a proxy explicitly injected
`output_config.effort=high`; one random fake-shell trajectory ran until the
diagnostic's 12-turn cap. This is a compatibility probe, not evidence of a
success-rate improvement. Its log is
[`wire-20522451.log`](../artifacts/claude-code-pass4-20260924/diagnostics/wire-20522451.log).

The 12-task Pi development baseline offers a closer reference than the old
v2.0 70-task result. Eleven task instructions match byte-for-byte by SHA256;
`kokkos__kokkos-8792` differs and is excluded. On those eleven tasks, Pi's
hosted Chat baseline passed **12/21 valid attempts**, whereas Claude Code
passed **5/44**. Both used the nominal hosted Inkling Small model ID, but the
API protocol, prompt construction, sample count, and possibly effective
sampling controls differ. This comparison establishes a serious observed
integration gap, not a causal harness estimate.

The exact split and counts are in
[`termination-analysis.json`](../artifacts/claude-code-pass4-20260924/diagnostics/termination-analysis.json).
The raw evaluation folders retain every CLI stream, shell event, candidate,
and grading record. The sampled wire response does not distinguish whether
the underlying model chose to end after thinking or Tinker's Anthropic renderer
converted its tokens into that terminal shape; that is the remaining protocol
boundary to investigate with Tinker.

The evaluator now supports a separate frozen `thinking_only_resume` profile.
It resumes the same Claude Code session after a thinking-only, blank successful
result, preserving the sandbox and cumulative turn, tool, and wall-time budgets.
The completed 400 outcomes remain untouched. The CLI stream omitted the
upstream `end_turn` field in a reproduced thinking-only case (job `20522657`),
so the profile accepts an absent stop reason only alongside the blank-success
signal. A forced-resume fake-MCP probe (job `20522688`) confirmed that
`--resume` preserved conversation history and produced two further tool calls;
it does not measure benchmark improvement. The logs are
[`wire-20522657.log`](../artifacts/claude-code-pass4-20260924/diagnostics/wire-20522657.log)
and [`wire-20522688.log`](../artifacts/claude-code-pass4-20260924/diagnostics/wire-20522688.log).
The new profile needs a separate scored evaluation before any efficacy claim.
