# Claude Code / Inkling Small: thinking-continuation 100-task evaluation

The bounded thinking-continuation profile substantially improved this Claude
Code integration on the frozen SWE-Kokkos-Bench set. On the **99 tasks with four
valid grades in both runs**, it solved **58/99 (58.6%) at pass@4**, versus
**25/99 (25.3%)** for the original one-shot profile. The paired difference is
**+33/99 tasks (+33.3 percentage points)**, with a task-bootstrap 95% interval
of **+24.2 to +42.4 points**. All 25 formerly solved tasks remained solved;
33 formerly unsolved tasks were solved. These are descriptive results of one
run per profile, not a measurement of the RFT checkpoint or a harness ranking.

All 400 scheduled attempts finished, but two candidates for
`kokkos__kokkos-8182` timed out in the hidden verifier before a score was
produced. That train/CPU task is **excluded from the final pass@4 denominator**:
the evaluation has 99 complete tasks, not a valid 100-task point estimate.
The observed all-task pass@4 lies between **58/100 and 59/100** depending on
the two ungraded attempts. The original run scored 25/100 with all grades
valid. No attempt or verifier was automatically rerun.

| Split | Four valid grades | Continuation pass@4 | Original on same tasks | Valid single attempts |
| --- | ---: | ---: | ---: | ---: |
| All | 99/100 | 58/99 (58.6%) | 25/99 (25.3%) | 162/398 (40.7%) |
| Train | 79/80 | 45/79 (57.0%) | 21/79 (26.6%) | 127/318 (39.9%) |
| Heldout | 20/20 | 13/20 (65.0%) | 4/20 (20.0%) | 35/80 (43.8%) |
| CPU | 95/96 | 55/95 (57.9%) | 23/95 (24.2%) | 152/382 (39.8%) |
| GPU | 4/4 | 3/4 (75.0%) | 2/4 (50.0%) | 10/16 (62.5%) |

Here empirical pass@4 means at least one passing attempt among four. The last
column is the fraction of **valid** attempts that passed, not a complete
400-attempt pass@1 point estimate. Across all 400 attempt slots, the two
ungraded outcomes bound pass@1 between **162/400 and 164/400** (40.5–41.0%).
On the 99 complete paired tasks, it was **162/396 (40.9%)**, versus
**47/396 (11.9%)** before: +29.0 points, with a task-bootstrap 95% interval
of +23.0 to +35.4 points. The task/sample labels match, but stochastic model
draws were not seed-coupled; sample-level transitions are descriptive.

## Execution and score integrity

The frozen [design](../artifacts/claude-code-pass4-resume-20260926/design.json)
and [CPU/GPU indices](../artifacts/claude-code-pass4-resume-20260926/cpu-index.json)
match the original run exactly for the 100 task IDs, 80/20 train/heldout
split, 96/4 CPU/GPU split, 400 task/sample slots, task and image digests,
hosted model ID, Claude Code package lock, and 40-model-turn / 80-tool-call /
one-hour episode budgets. The new frozen policy is `thinking_only_resume`,
with at most three neutral continuations in the same Claude Code session and
sandbox. Freeze job `20523331`, CPU/GPU smoke jobs `20523332`/`20523333`, and
all 384 CPU plus 16 GPU formal array elements completed with Slurm exit `0:0`
([saved accounting](../artifacts/claude-code-pass4-resume-20260926/slurm-sacct.txt)).

Each of the 400 formal folders has a preflight, candidate, summary, status,
grading record, model events, stdout/stderr, and reported usage. All 400
preflights confirmed hidden assets were absent from the sandbox; bridge error
count was zero. The verifier returned a score on **398** attempts: 162 pass,
236 valid failure. The other two are the only infrastructure failures; no
attempt is missing. The locally recomputed [analysis](../artifacts/claude-code-pass4-resume-20260926/analysis.json)
exactly equals the [Rivanna analysis](../artifacts/claude-code-pass4-resume-20260926/analysis-rivanna.json).

For `kokkos__kokkos-8182` samples 0 and 2, the candidate patches were saved,
but the verifier command `bash /tests/test.sh` returned `exit=-1` after its
frozen 900-second timeout, before producing a score. The task's other two
attempts had valid reward zero. The failing verifier rebuilds the serial
containers target; the candidate patches touch different production headers.
The exact reason for the two build timeouts is unproven. Original grading,
patch, status, and logs remain in the [sample 0](../artifacts/claude-code-pass4-resume-20260926/results/kokkos__kokkos-8182/sample-0)
and [sample 2](../artifacts/claude-code-pass4-resume-20260926/results/kokkos__kokkos-8182/sample-2)
folders. Their recorded reward zeros are **not** treated as scored model
failures. Adjudicating this one task would require a separate, explicitly
recorded verifier investigation; it is not needed to see that the paired
99-task improvement is large.

## Continuation behavior and cost

The original adapter often accepted a thinking-only `end_turn` as a blank
successful completion. The new profile resumed **376/400** episodes at least
once; 305 used all four invocations (original plus three continuations), 40
used three, 31 used two, and 24 used one. It produced 158 passing attempts
among resumed episodes. The final candidates changed production source in
**241/400** attempts, including the two ungraded ones; the original run did
so in 67/400. On the strictly paired 99 tasks, source-edit attempts rose from
**66/396 to 237/396**. These counts describe final candidate diffs: they do
not prove the edit happened after the first invocation, or attribute every
passing attempt causally to continuation.

| Paired 396 valid attempts | Original | Continuation |
| --- | ---: | ---: |
| Median episode time | 53.0 s | 105.7 s |
| Median model turns | 7 | 24 |
| Median executed MCP calls | 6 | 20 |
| Client-reported input tokens, summed | 23,531,499 | 103,042,677 |
| Client-reported output tokens, summed | 528,738 | 1,819,830 |

The [full-run metrics](../artifacts/claude-code-pass4-resume-20260926/metrics.json)
record 8,563 executed MCP calls across 398 valid attempts, median 20 per
attempt, and 18.27 summed episode-hours; the original 400 valid attempts used
3,079 calls and 8.72 summed hours. Summed episode-hours are not cluster
makespan because jobs ran concurrently. Reported input totals add Claude
Code's input, cache-creation, and cache-read counters; these are client
telemetry, not verified cache hits or Tinker billing. Thirty-nine valid
episodes hit the model-turn budget; nine still passed using their captured
candidate. The event counter observed 41 assistant messages in those cases
because the guard stops the process immediately after receiving the message
that crosses the 40-turn threshold. Token totals for process-terminated
invocations may be lower bounds because a final CLI usage record was not
always emitted.

Continuation improved usable coding behavior but did not fully resolve the
protocol mismatch. Of 400 final assistant events, 239 were thinking-only,
123 text, and 38 tool-use; 60 final text events still contained MCP-call
syntax as text. The profile reached its three-continuation limit in 235 of
the 396 fully scored paired attempts. The next engineering target is a
properly normalized tool/stop protocol and a budget-matched comparison, not
simply increasing continuation count.

## Interpretation and reproduction

This is the hosted `thinkingmachines/Inkling-Small:peft:262144` **Chat**
weight path through Tinker's Anthropic-compatible API, Claude Code `2.1.50`,
and the isolated HPC MCP shell. It is not the trained Inkling Small RFT/RL
checkpoint. Its inferred gain is specific to fixing the thinking-only stop
behavior in this integration; it does not establish that Claude Code beats Pi,
Codex, DeepSeek, or the historical native agent. Those paths use different
prompts, protocol rendering, or inference controls. The 20-task heldout and
four-task GPU strata are small, and the task-bootstrap interval reflects only
resampling these task outcomes, not uncertainty over model versions or harness
configuration. Skills, todo tools, built-in tools, and project settings were
disabled in both Claude Code profiles, so this run does not assess their value.

The [paired comparison](../artifacts/claude-code-pass4-resume-20260926/paired-comparison.json)
contains task and attempt transitions and the bootstrap calculation. Recompute
locally from the workspace root with `python3 harness-compare/analyze_full.py
artifacts/claude-code-pass4-resume-20260926`, `python3
harness-compare/summarize_full.py
artifacts/claude-code-pass4-resume-20260926`, and `python3
harness-compare/compare_resume.py artifacts/claude-code-pass4-20260924
artifacts/claude-code-pass4-resume-20260926`. The complete [400 result
folders](../artifacts/claude-code-pass4-resume-20260926/results),
[Slurm logs](../artifacts/claude-code-pass4-resume-20260926/slurm), and
[submission record](resume-full100-submission.json) are retained.
