# Claude Code / Inkling Small: SWE-Kokkos-Bench 100-task pass@4

All 400 frozen Rivanna attempts finished with valid grading. Claude Code solved
**25/100 tasks at pass@4 (25.0%)**; the empirical single-attempt rate was
**47/400 (11.75%)**. There were no missing attempts or infrastructure failures.
This run used the hosted `thinkingmachines/Inkling-Small:peft:262144` Chat
weights, recorded in the design as `hosted_chat_not_rft_checkpoint`. It is a
Claude Code harness baseline, not an evaluation of the HPC RFT checkpoint or
the Pi HPC skill.
Post-run [termination diagnosis](TERMINATION_DIAGNOSIS.md) found that all 400
CLI sessions ended with either a thinking-only response or unexecuted MCP-call
syntax in text. The score is valid for the candidates produced by this
configuration, but the integration is not a healthy baseline for ranking
harnesses.

| Split | Tasks with four valid scores | pass@4: at least one of four | pass@1: passing attempts / attempts |
| --- | ---: | ---: | ---: |
| All | 100/100 | 25/100 (25.0%) | 47/400 (11.75%) |
| Train | 80/80 | 21/80 (26.25%) | 41/320 (12.81%) |
| Heldout | 20/20 | 4/20 (20.0%) | 6/80 (7.50%) |
| CPU | 96/96 | 23/96 (23.96%) | 44/384 (11.46%) |
| GPU | 4/4 | 2/4 (50.0%) | 3/16 (18.75%) |

Here pass@1 is the fraction of the four independently scored attempts that
passed. Seventy-five tasks had no passing attempt; among the 25 solved tasks,
10 passed once, 9 twice, 5 three times, and 1 all four times. The four GPU
tasks are too few to support a separate performance conclusion. The train and
heldout percentages are descriptive; the 20-task heldout group is too small
to establish a reliable gap.

## Integrity and execution

The successful freeze job `20462014` verified 100 task payloads and image
hashes against the validated `inkling-small-round02` source, including all
200 NOP/oracle checks. Its design fixes the 80/20 train/heldout split, 96/4
CPU/GPU task allocation, four samples per task, and 400 distinct task/sample
indices. The earlier freeze attempt `20461832` exposed an overly strict
Dockerfile provenance check for four GPU runtime images rebuilt for Rivanna.
The corrected freeze verifies the original Dockerfiles against the source
manifest, and verifies the runtime SIF hashes and GPU NOP/oracle results.

CPU smoke job `20462229` and GPU smoke job `20462230` each produced a passing
preflight, MCP shell calls, a candidate artifact, and a verifier record. The
full CPU array `20462231` finished 384/384 elements and the GPU array
`20462232` finished 16/16; all 400 Slurm elements exited `0:0`. Every full
attempt has a summary, status, preflight, candidate, grading output, model
events, stdout, stderr, and CLI usage record. All 400 preflights kept hidden
assets out of the sandbox, candidate captures completed, graders exited zero,
and neither Claude Code nor the MCP bridge reported an error. The 353 reward-0
episodes are verifier failures, not missing or infrastructure-failed runs.

Local `analyze_full.py` output exactly matches the analysis saved on Rivanna.
All 400 attempts are valid, so the final pass@4 denominator contains every
task. The smoke verifier and analysis tests pass locally. The source and
evaluator hashes match the frozen design.

## Budget and usage

| Measure across 400 valid attempts | Value |
| --- | ---: |
| Attempts stopped at the 40-model-turn cap | 0 |
| Largest observed model-turn count / 40 cap | 33 |
| Largest executed sandbox-call count / 80 cap | 32 |
| Median model turns / executed sandbox calls | 7 / 6 |
| Total executed sandbox calls | 3,079 |
| Median episode elapsed time | 53.1 s |
| Sum of episode elapsed times | 8.72 h |
| Median client-reported input / output tokens | 29,788 / 643 |
| Sum of client-reported input / output tokens | 23,871,076 / 538,089 |

Episode time includes sandbox setup, model interaction, and grading; the sum
is not cluster makespan because jobs ran concurrently. Input tokens above add
Claude Code's `input_tokens`, `cache_creation_input_tokens`, and
`cache_read_input_tokens`. The [Tinker Anthropic API documentation](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/)
says prompt caching is ignored by this beta endpoint, even though Claude Code
reported nonzero cache-related fields. These figures are client telemetry,
not verified cache hits or Tinker billing. No trajectory crossed the frozen
40-turn or 80-tool-call budgets, so no budget-truncated score enters pass@4.

## Observed harness behavior

Sixty-seven candidates changed production source: 47 passed and 20 failed.
The other 333 made no production-source change and all failed. The candidate
artifact also contains sandbox `workspace/bash_events` files, so merely
finding a nonempty `candidate.patch` does not prove a code edit. None of the
passing candidates modified test or grading paths.

In 379/400 trajectories, Claude Code's final assistant event contained only
a thinking block and no final answer text; this also occurred in passing
trajectories. The other 21 final text events contained unexecuted MCP-call
syntax (20 with the complete `mcp__hpc__bash` name and one with a truncated
name). A wire-level reproduction shows Tinker's Anthropic endpoint returning
`thinking` alone with `stop_reason=end_turn`, which Claude Code treated as a
successful, blank completion. This is a material integration defect for this
harness/model path; see the [termination diagnosis](TERMINATION_DIAGNOSIS.md).
One additional attempted MCP call used `cmd` instead of the tool's required
`command` field and was rejected by CLI schema validation; 3,079 calls reached
the sandbox. These are useful diagnostics for a future harness iteration, but
the present run's candidates and verifier outcomes remain valid. The profile
disabled Claude Code's built-in tools, slash commands, user/project settings,
skills, and todo tools; only `mcp__hpc__bash` was exposed. Skill and todo use
was therefore zero by configuration, not evidence that those features lack
benefit.

## Interpretation and evidence

The historical native terminal-agent baseline for the 70 tasks added in
SWE-Kokkos-Bench v2.0 passed **42/70 single attempts (60.0%)** with the nominally
same hosted Inkling Small model ID. On those same 70 task IDs, this Claude Code
run passed **34/280 attempts (12.14%)** and solved **16/70 tasks within four
attempts (22.86%)**. The 30 older tasks account for the remaining 13/120
passing attempts and 9/30 tasks solved in this run. This is a large observed
regression against the historical native scaffold, despite giving Claude Code
four chances per task. It is not an isolated measurement of the harness effect:
the native baseline used v2.0 task instructions and native Tinker rendering with
`thinking_effort=0.9`, while this run used the later frozen task instructions,
the Anthropic-compatible route without an explicit Tinker effort setting, and
Claude Code's own prompt/tool protocol. The historical baseline has only one
sample per task and no complete 100-task Inkling Small score. See the
[benchmark baseline table](../tinker-cookbook/data/kokkos/SWE-kokkos-bench-v2/README.md)
for its denominator and settings.

This score describes one frozen 100-task development/heldout split, one hosted
Chat model path, and Claude Code `2.1.50` with the isolated MCP shell bridge.
It does not estimate the released or RFT-trained Inkling Small model's HPC
capability. The Pi pilot used Tinker's Chat Completions route with numeric
`reasoning_effort=0.9` and `separate_reasoning=true`; this Claude Code run used
the Anthropic route and did not set Tinker's `output_config.effort`. Prompt
construction, sampling controls, and replication are not matched, so the
[four-task cross-harness pilot](REPORT.md) and this full run cannot establish
a harness ranking. A controlled comparison needs the same 100 tasks, four
attempts per task, hosted weights, inference settings, and request/tool/time
budgets across harnesses.

Recompute the results with `python3 harness-compare/analyze_full.py
artifacts/claude-code-pass4-20260924` and `python3
harness-compare/summarize_full.py
artifacts/claude-code-pass4-20260924` from the workspace root. Evidence is in
the frozen [design and indices](../artifacts/claude-code-pass4-20260924/design.json),
[attempt-level analysis](../artifacts/claude-code-pass4-20260924/analysis.json),
[compact metrics](../artifacts/claude-code-pass4-20260924/metrics.json),
[400 result folders](../artifacts/claude-code-pass4-20260924/results), and
[Slurm records](../artifacts/claude-code-pass4-20260924/slurm).
