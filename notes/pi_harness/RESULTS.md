# Pi harness iteration for Inkling-Small on SWE-kokkos-bench v2.4

Goal: a Pi harness configuration whose Inkling-Small pass@1 is significantly above the
same-period native loop on Rivanna (combined across blocks, p < 0.05), with every change
motivated by trajectories, opt-in, and guarded by tests. Episodes run through
`notes/dsh_inkling/harness/rivanna/block.py` with arm kind `pi`; the agent is
[Luosuu/pi-hpc-agent](https://github.com/Luosuu/pi-hpc-agent).

## Parity first (pi-hpc-agent 0.2.0)

Before any lever, the Pi arm was made to show the model exactly what the native loop shows:

- Pi appended `<cwd>/tmp/kokkos-pi-…</cwd>` (the host session directory) to the system
  prompt. It is now removed; skill-mode `<skills>` stays.
- Bash results were `JSON.stringify` output with no overall cap. They are now rendered like
  Python `json.dumps` and truncated to 16384 characters keeping the first third and last two
  thirds, exactly as `eval_kokkos_openai`. A cross-language test
  (`eval_kokkos_pi_test.py::test_pi_tool_result_is_byte_identical_to_the_native_loop`)
  caught one divergence during development: Python escapes DEL (`\x7f`).
- The bash tool description matches the native schema. Prior assistant reasoning is sent
  back by both loops (`reasoning_content`), so history handling already matched.

## What the native trajectories say (block r01, 399 native episodes)

`python -m notes.pi_harness.taxonomy BLOCK`:

| native failure (179) | count |
| --- | ---: |
| grader build failed | 78 |
| no source change | 74 |
| fail-to-pass test failed | 22 |
| pass-to-pass test failed | 3 |

- **The request cap binds.** 157 of 179 failures (88%) ended at the 40-request cap; the
  median failed episode used all 40. 91 of 220 solves also ended at the cap.
- **One command per request.** 14,155 of 14,155 tool-calling responses had exactly one call.
  The 80-tool-call budget never binds.
- **Reads dominate.** Commands in failed episodes were 75% (build-failed) and 89%
  (no-change) reads. Reads are small: median 520-620 characters (3-4% of the 16384 budget),
  `sed -n` windows of 41-61 lines, and failed episodes re-read the same file 3.6 times.
- **Solves edit earlier.** First edit at command 14 (solved) against 26-28 (failed); within
  38 mixed tasks, solved draws edited earlier in 24.
- **Most build failures were visible.** Of 230 grader build failures across all three arms,
  the first error was in a file the model edited (96), another existing source or test file
  (49), or a hidden test file the model never sees (76). About 63% would have shown up in
  the model's own build of the grader's visible target.
- Task outcomes: 27 always solved, 21 never, 52 mixed.

## Levers

### batch-inspect guidance (0.3.0): rejected before a block

Asks the model to combine independent inspection steps into one bash command. In a
one-shot probe it packed 5-8 steps into the first command (baseline: 1), and parallel tool
calls were never emitted even when requested. In three full episodes the effect vanished:
first-third commands averaged 1.0-1.3 parts, the same as the Pi baseline. A system-prompt
sentence does not change the model's working habit. Not run as a block arm.

### Sandbox read/edit tools (0.4.0): weak mechanism

Pi's own edit tool and a read tool with Pi's schema, paged at 16384 characters, executed
in the sandbox. In six smoke episodes the model used them (25 reads, 23 edits, 9% edit
errors) but kept bash for most work (184 calls); median read size rose only to ~2000
characters, first edit stayed at call 29, and 5 of 6 hit the request cap. Kept as an
option; not a primary arm.

### Harness build check after edits (0.5.0)

Targets the visible build failures and the unbuilt final edits: after any bash or edit call
that changes the source tree, the harness compiles the grader's visible target and appends
a factual summary with the first errors to that tool result. It costs wall time, not
requests or tool calls (at most 8 checks, 300 s each, 1200 s total). Grader build times in
r01: median 18 s, p75 163 s, p90 239 s.

Unlike the DSH managed checkpoint (call 32 or later, at most twice, fired in 29% of
episodes), it runs after every source change.
