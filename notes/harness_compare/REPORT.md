# Cross-harness SWE-Kokkos-Bench pilot — 2026-09-24

The separate [100-task Claude Code pass@4 report](FULL100_REPORT.md) supersedes
this four-task pilot for its absolute Claude Code score. This pilot remains a
small feasibility comparison and is not a matched full-benchmark harness rank.

Tinker's [Anthropic-compatible API](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/) let Claude Code `2.1.50` run the same hosted `thinkingmachines/Inkling-Small:peft:262144` model used by the Pi and DeepSeek Harness pilots. Claude Code completed all four frozen Rivanna development tasks. It passed **2/4** hidden verifiers; DeepSeek Harness `0.1.5-rc.3` passed **1/4**; Pi's earlier baseline passed **3/8** across two samples for each task.

| Task | Pi baseline, two samples | DeepSeek reward; model steps / tools | Claude Code reward; turns / tools |
| --- | ---: | ---: | ---: |
| `kokkos__kokkos-8594` | 1, 1 | 1; 19 / 18 | 1; 9 / 8 |
| `kokkos__kokkos-9276` | 0, 0 | 0; 76 / 75 | 0; 8 / 7 |
| `kokkos__kokkos-7172` | 1, 0 | 0; 37 / 36 | 1; 10 / 9 |
| `kokkos__kokkos-kernels-2864` | 0, 0 | 0; 51 / 50 | 0; 9 / 8 |

Claude Code's per-task reward minus the mean of Pi's two samples ranges from **0 to +0.5**. Its sole advantage is `7172`, where Pi also passed one of its two attempts. DeepSeek's corresponding range is **−0.5 to 0**. These four development tasks, with unequal replication and different inference settings, do not establish a general harness ranking. All Claude Code turns are below Pi's 40-model-request cap, although the CLIs' turn counts are not an exact shared request-budget measure. DeepSeek's `9276` and `kernels-2864` traces crossed that cap; their final failures are real verifier failures, but are not budget-matched comparisons. No intermediate step-40 candidate was graded.

| Harness | Median elapsed time | Median reported input / output tokens | Scope |
| --- | ---: | ---: | --- |
| Pi baseline | 119 s | 406.4k / 5.2k | Eight traces, two per task |
| DeepSeek Harness | 1,495 s | 570.5k / 6.6k | Four traces |
| Claude Code | 66 s | 45.8k / 0.7k | Four traces |

The Claude Code trajectories took 56–235 seconds each and used 7–9 sandbox calls. Its individual reported input/output counts were `41.3k/0.8k`, `39.3k/0.6k`, `50.4k/2.9k`, and `66.8k/0.6k` in table order. The input figures add Claude Code's `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`. Tinker's documentation says prompt caching is ignored by this beta endpoint, while Claude Code reports nonzero values in those cache fields. Therefore, treat these as **client-reported telemetry**, not verified cache hits or reconciled Tinker billing. Token counts across Chat Completions and Anthropic Messages may have different accounting. DeepSeek response stalls also affected its wall time, so the elapsed-time gap cannot be attributed to harness quality alone.

Each run used the original task instruction, validated SIF snapshot, network-isolated Apptainer sandbox, and Harbor hidden verifier. The Claude Code profile kept the model credential on the host, disabled built-in tools and user/project settings, and exposed only `mcp__hpc__bash` through a Unix-socket bridge to `SandboxInterface.run_command`. Every Claude run produced a candidate patch, grading result, and CLI result event; all four Slurm jobs exited 0, and the bridge recorded no errors. Rewards of zero are verifier failures, not missing or infrastructure-failed trajectories. Its two passing traces also had successful verifier grading. Skill and todo use is zero in this baseline by configuration, so this pilot does not measure the HPC skill's transfer to Claude Code.

Pi used numeric `reasoning_effort=0.9` and `separate_reasoning=true` on the Chat Completions route. The stock DeepSeek provider and Claude Code CLI did not pass those same Tinker-specific fields; Claude Code also supplied its own system prompt and Anthropic Messages rendering. The comparison therefore matches weights and tasks, but not inference parameters or prompt construction. The results apply to this small development subset and hosted Inkling Small weights, not a release model or the whole benchmark. A controlled follow-up needs the same sample count and explicit request, token, tool, and wall-time budgets.

Codex is still absent from the score table. Its [custom model providers](https://learn.chatgpt.com/docs/config-file/config-reference) use Responses, while the available Tinker routes are Chat Completions and Anthropic Messages. A same-Inkling Codex run still needs a validated Responses adapter; a native OpenAI-model run would measure model and harness together.

Evidence: [`analysis.json`](../artifacts/harness-cross-eval-20260924/analysis.json), [`claude-results`](../artifacts/harness-cross-eval-20260924/claude-results), and the existing DeepSeek result directories contain summaries, candidate patches, grading outputs, sandbox command events, and model telemetry. Slurm jobs `20461077` and `20461117` completed with exit code 0; the DeepSeek jobs were `20459113` and `20459291`.
