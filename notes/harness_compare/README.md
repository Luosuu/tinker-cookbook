# Cross-harness SWE-Kokkos-Bench evaluation

This pilot uses the frozen Rivanna task images and Harbor verifier from the Pi
study. The DeepSeek Harness profile runs on the host for model access; its only
repository tool is a socket bridge to `SandboxInterface.run_command`, so source
edits and builds remain inside the original network-isolated Apptainer sandbox.
No model credential is passed into the sandbox.

The DeepSeek profile pins `@deepseek-ai/dsh@0.1.5-rc.3` and targets the same hosted
Inkling Small Chat Completions route as Pi. The patch removes all ordinary
filesystem, shell, web, and delegation tools and adds one `bash` tool that
calls the sandbox bridge. The original user instruction, validated SIF, network
isolation, candidate capture, and Harbor hidden tests are reused.
`eval-initial.sbatch` records the first two tasks; `eval.sbatch` records the
additional two. `submission.json` records the Slurm IDs.
For a fresh Rivanna install, copy this directory to the run root, run
`npm ci --ignore-scripts --no-audit --no-fund` in `dsh-runtime`, and submit
the Slurm scripts after setting `HPC_PI_EVAL_ROOT` to a validated task snapshot.

The Claude Code profile pins `@anthropic-ai/claude-code@2.1.50` in
`claude-runtime`. Install it with the same `npm ci` command in that directory,
and provide Python `mcp==2.2.0` in the evaluator venv before submitting
`eval-claude.sbatch`. It points `ANTHROPIC_BASE_URL` at Tinker's
Anthropic-compatible endpoint and maps `TINKER_API_KEY` to `ANTHROPIC_API_KEY`
on the host. Built-in tools and user/project settings are disabled; the only
exposed repository tool is `mcp__hpc__bash`, implemented in `mcp_hpc.py` and
bridged to the same isolated sandbox. The credential is never passed into
the sandbox or MCP config. Each result directory captures CLI events, sandbox
commands, candidate patch, and verifier output. The four-task pilot used
Slurm jobs `20461077` and `20461117`.
The standalone `probe_anthropic.py` HTTP client received a 403/1010 response
from the beta endpoint on both login and compute nodes; `probe_claude.py` and
the full Claude Code runs succeeded from Rivanna. The raw HTTP probe is not a
reliable reachability check for this client configuration.

| Harness | Same Inkling Small endpoint | Current state |
| --- | --- | --- |
| Pi | Native `openai-completions` model provider | 12 tasks × 2 samples from the September 22 study |
| DeepSeek Harness | Native `openai-completions` custom provider | Rivanna pilot, `0.1.5-rc.3` |
| Claude Code | Tinker Anthropic-compatible endpoint | Direct same-Inkling route; Rivanna tool-routing smoke passed |
| Codex CLI/SDK | Custom providers require Responses API; Tinker exposes Chat Completions | Needs a vetted Responses adapter, or an OpenAI model credential for a separate model-plus-harness study |

The compatibility facts come from the [DeepSeek provider guide](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.md), [Tinker Anthropic-compatible API guide](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/), and [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

The DeepSeek pilot is a feasibility check, not a significance test. Pi's
`reasoning_effort` is a numeric 0.9 with `separate_reasoning: true`; the stock
DeepSeek provider does not expose those Tinker-specific request fields, so
model identity is matched but inference parameters are not yet identical.
The bridge caps tool calls at 80 and the process at one hour, while the stock
headless runner does not expose Pi's 40-model-request cap. A DeepSeek solution
after 40 model steps is therefore an exploratory success, not a budget-matched
win. Report these caveats with every score comparison.

## Full Claude Code pass@4 run

`freeze_full.py` copies the validated September 2026 100-task snapshot into a
new experiment directory. It checks the 80/20 train/heldout split, all 200
NOP/oracle records, the task payloads, and the SHA-256 of every SIF. The frozen
index contains four independent samples per task: 384 CPU and 16 GPU episodes.
`eval-full-cpu.sbatch` uses the CPU partition at four concurrent episodes;
`eval-full-gpu.sbatch` uses the same MIG GPU allocation that validated the
four CUDA images, at one concurrent episode. Each output has its own sandbox,
Claude Code home, candidate patch, grading result, command events, CLI result,
and status. If Claude Code emits a 41st model turn, the evaluator stops it
before another tool action, grades the candidate from the preceding turns,
and records the budget stop separately from infrastructure errors. The extra
response may already have been sampled, so billed requests are not exactly
matched to Pi's 40-request cap.

Run `analyze_full.py <experiment-root>` during or after evaluation. A task
contributes to pass@4 only after all four samples have valid grading; missing
and infrastructure-failed samples remain visible, with provisional lower and
upper bounds. Train and heldout scores are reported separately. The run uses
the baseline Claude Code profile, with no HPC skill or todo tool injected.

The completed 100-task run is summarized in [FULL100_REPORT.md](FULL100_REPORT.md).
The [termination diagnosis](TERMINATION_DIAGNOSIS.md) documents the
thinking-only `end_turn` response reproduced at the Anthropic interface and
the completion-quality field now recorded by future frozen evaluators.
Its attempt-level evidence is under
`../artifacts/claude-code-pass4-20260924/`. Run
`summarize_full.py <experiment-root>` to reproduce the split, usage, and
candidate-edit metrics alongside `analyze_full.py`.

## Bounded thinking-only continuation profile

For a new frozen experiment, `freeze_full.py` accepts two optional positional
arguments after the runtime path: `thinking_only_resume 3`. This records a
separate continuation policy in `design.json`; the completed 400-attempt run
above remains the original one-shot baseline. A `thinking_only` assistant
message and a blank successful CLI result trigger a neutral
`--resume` prompt in the same Claude Code home, MCP bridge, and Apptainer
sandbox. Claude Code omits the upstream `end_turn` reason in its stream event,
so the adapter accepts an absent stop reason only with a blank successful CLI
result. Final text and other completion types do not trigger a resume.
The episode retains one shared 40-model-turn, 80-tool-call, and one-hour
budget, with at most three continuations. `continuations.jsonl` records every
invocation, and `status.json` records the final stop reason. `claude_result.json`
retains per-invocation token usage and their sum. Exhausting the continuation
limit leaves the resulting
candidate for normal grading, while a CLI resume error is an infrastructure
failure. This profile must be evaluated as a new harness variant before making
any score claim.
