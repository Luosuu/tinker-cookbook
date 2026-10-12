# Pi harness iteration for Inkling-Small on SWE-kokkos-bench v2.4

Goal: a Pi harness configuration whose Inkling-Small pass@1 is significantly above the
same-period native loop on Rivanna (combined across blocks, p < 0.05), with every change
motivated by trajectories, opt-in, and guarded by tests. Episodes run through
`notes/dsh_inkling/harness/rivanna/block.py` with arm kind `pi`; the agent is
[Luosuu/pi-hpc-agent](https://github.com/Luosuu/pi-hpc-agent).

## Result

**Pi with the harness build check (`build_check=after-edit`, pi-hpc-agent ≥ 0.5.0) beats the
same-period native loop by 4.3 points pass@1 over two blocks (task-paired t-test p = 0.010,
task bootstrap 95% interval +1.3 to +7.5; 1,197 of 1,200 block-p01 episodes and 1,193 of
1,200 block-p02).** The p02 run was the pre-specified replication of the p01 lead and reached
significance on its own (task-paired p = 0.008; draw-level Welch t, p = 0.021). The Pi loop without levers matches
native (−1.2, p = 0.52). The symbol brief adds nothing on top of the build check (+0.5,
p = 0.84).

| arm | p01 draws | p02 draws | pass@1 over both |
| --- | --- | --- | ---: |
| native | 58.0, 52.0, 57.0, 55.0 | 49.0, 51.5, 54.5, 55.6 | 54.1% |
| Pi | 54.0, 54.5, 60.6, 49.0 | — | 54.5% (p01) |
| Pi + build check | 62.6, 56.6, 56.0, 57.0 | 58.0, 62.0, 57.0, 57.6 | 58.4% |
| Pi + build check + symbol brief | — | 59.6, 54.5, 64.0, 56.6 | 58.7% (p02) |

`python -m notes.pi_harness.combine --treatment pi_check --control native BLOCK_p01 BLOCK_p02`
reproduces the primary test. Robustness: excluding the systemic verifier timeout
(`kokkos__kokkos-8182`) gives +4.3, p = 0.010; the draw-level inverse-variance combination
gives +4.4 (SE 1.4). With two Pi arms compared against native in p02, a Bonferroni-adjusted
primary p is 0.02. The unfinished episodes (three in p01, seven in p02) are all
`kokkos__kokkos-8182`, spread across the arms.

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

## Block p01: native vs Pi vs Pi + build check

`dsh-blocks/block-p01`, code `0efdb4b`, pi-hpc-agent 0.5.0, four draws of all 100 tasks per
arm, 36 concurrent 4-CPU/16 GB episodes, arm order interleaved task by task. 1,190 of 1,200
episodes had finished when recorded (the rest are long-running); zero infra errors.

| arm | draws | pass@1 | pass@2 |
| --- | --- | ---: | ---: |
| native | 58.0, 52.0, 57.0, 55.1 | 55.5% | 64.8% |
| Pi | 54.0, 54.5, 60.6, 49.0 | 54.2% | 62.7% |
| Pi + build check | 62.6, 56.6, 56.6, 56.7 | 58.2% | 67.0% |

| comparison | paired pass@1 (95% CI) | draw-level delta, p |
| --- | --- | --- |
| Pi vs native | −1.2 (−5.0 to +2.5) | −1.0, p = 0.72 |
| Pi + check vs native | +2.1 (−1.9 to +6.1) | +2.6, p = 0.20 |
| Pi + check vs Pi | +3.4 (−1.4 to +8.1) | +3.6, p = 0.20 |

**Parity holds:** with the 0.2.0 fixes the Pi loop scores like native, as DSH did.

**The build check moves its mechanism:** grader build failures fell from 85 (Pi) to 51.
The model's own habit did not change: failures whose last edit the model never built itself
stayed at 77 (Pi) and 80 (Pi + check); the harness builds for it. Part of the gain moved into
compiled-but-wrong patches (fail-to-pass failures 24 → 31). (An earlier count of 43 included
the harness's own build commands as the model's.) No-source-change failures (67 → 79) are draw
noise, not reverts: 77 of the 79 never triggered a check and only 2 used `git checkout`.

## Correction to the identifier-search count

An earlier count reported 8-24 pre-edit searches per episode for instruction identifiers.
It matched the boilerplate `/workspace/repo` path, which every command contains. With
identifiers taken from the task prose (scoped names, snake/camel case, calls), the counts are
3.2 (solved), 4.8 (build failed), 6.5 (fail-to-pass failed) and 8.8 (no source change) of
9-26 pre-edit searches.

### Symbol brief (0.6.0)

Before turn 1 the harness greps the task's identifiers over source files and delivers the
locations once, as a separate message: rarest identifiers first, definitions before uses,
ubiquitous names only listed. For kokkos-6289 it lists the two functions the task removes
and every call site to change.

## Block p02: replication, and the symbol brief on top

`dsh-blocks/block-p02`, code `bd91a5e`, pi-hpc-agent 0.6.0, same envelope as p01. 1,193 of
1,200 episodes finished; zero infra errors (one native episode was OOM-killed during sandbox
teardown after it had been graded).

| comparison (p02) | task-paired delta (95% CI) | p |
| --- | --- | ---: |
| Pi + check vs native | +6.25 (+2.0 to +10.75); draw-level +6.0 | 0.008; Welch t 0.021 |
| Pi + check + brief vs native | +6.75 (+1.0 to +12.75) | 0.023 |
| Pi + check + brief vs Pi + check | +0.5 (−4.5 to +5.5) | 0.84 |

| failures (p02) | native | Pi + check | + brief |
| --- | ---: | ---: | ---: |
| grader build failed | 86 | 61 | 67 |
| no source change | 76 | 70 | 57 |
| fail-to-pass failed | 21 | 30 | 33 |

The build check again cut grader build failures by about 30% against native. The brief
lowered no-change failures (70 → 57) but those episodes then failed later, mostly at
fail-to-pass and build, so the score did not move. In the six-task smoke it moved the
first edit from command 33 to 28.5 and pre-edit searches from 20 to 15.5. It stays
opt-in.

Hidden-test compile failures that name a task identifier are only 23 of 163 (14%), so a
"named identifier still missing" signal was not built.

## Recommended configuration

`build_check=after-edit` with the matched native budget (40 requests, 80 tool calls,
16384-character tool results). It compiles the grader's visible target after source
changes, costs wall time only (median grader build 18 s; at most 8 checks, 1200 s), and
needs `build_check_command`, which the cookbook evaluator derives from the task verifier.

## What did not work

- batch-inspect guidance: no change in working habit over full episodes.
- Sandbox read/edit tools: adopted for a minority of reads and edits, little more text per
  read.
- Symbol brief on top of the build check: shifts failures, not the score.

## At a 250-request budget

`notes/pi_harness/BUDGET.md`: in one block, native at 250 requests beat native at 40 by
+12.1 points (p = 0.001), and Pi with the build check at 250 was only +1.6 over native at 250
(p = 0.55). The build check's gain is specific to the frozen 40-request budget.
