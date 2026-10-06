# DSH harness results on SWE-kokkos-bench

Model: `thinkingmachines/Inkling-Small:peft:262144`. Budget: 40 model requests, 80 tool
calls, 65,536 sampled tokens, temperature 1, `reasoning_effort` 0.9 with
`separate_reasoning`, network-disabled sandbox. Metric: pass@1 over all 100 frozen tasks.

## Headline

| Run | Harness | Sandbox | Solved | vs reference |
| --- | --- | --- | ---: | --- |
| reference | native Chat Completions tool loop | Rivanna Apptainer | **52/100** | — |
| `full01` | DSH | ConTree | 47/100 | −5 |
| `full02` | DSH + budget/fidelity fixes | ConTree | **53/100** | **+1** |

**The +1 is a tie, not a demonstrated improvement.** 23 discordant pairs (12 gained, 11
lost), exact McNemar two-sided p = 1.000; the 95% interval on the difference spans roughly
−8 to +10 points. Report it as parity with a working iteration platform, not as a win.

## What the runs establish

### The comparison is sound

Audited on all 100 episodes of `full02`:

- **Prompt parity**: every episode sends a 164-character system message, exactly one user
  message, and `bash` as the only tool — the reference shape.
- **Budget parity**: 3,592 upstream requests, of which 3,592 are task turns and **0 are
  housekeeping**. No request is spent outside the task budget.
- **Token accounting live**: no episode records requests while reporting zero input
  tokens, so the input-token budget is genuinely enforced.
- **Grading at the cap**: episodes that end at the 40-request cap are still graded on the
  candidate in the sandbox, matching the reference's cap semantics. 23 of 53 solves needed
  the full budget.

### Where the remaining headroom is

| group | n | pass@1 |
| --- | ---: | ---: |
| ended at the 40-request cap | 62 | 32.3% |
| finished on their own | 38 | **86.8%** |
| capped, never attempted a write | 11 | 0% |

The model is capable when it stops on its own. The loss is concentrated in episodes that
run out of budget, and 11 episodes never edit at all. Command mix across 3,552 calls:
77.4% read/search, 11.7% build, 10.9% other.

## Defects found and fixed along the way

Each of these was an infrastructure or fidelity fault that masqueraded as model failure.
They are why the first full run scored 47 rather than 53.

1. **DSH could not send the reference inference parameters.** Its pi-ai provider selects a
   reasoning level by name only; a capture-server check confirmed it emitted neither
   `reasoning_effort` nor `separate_reasoning`. A local proxy injects both (plus
   temperature 1) on every upstream request, so comparability is structural rather than
   configurable.
2. **The harness had no model-request cap.** `dsh --profile headless` exposes none, so the
   proxy enforces the 40-request and 5M-input-token budgets.
3. **Three prompt differences.** DSH prepended its own identity line and a "Current runtime
   context" user message; `plan-mode` injected a "stay in plan mode, do not edit" section
   plus an extra tool; `agent-instructions` added another user message. All are disabled.
4. **A non-zero harness exit skipped grading**, discarding a patch that already existed.
   The cap makes the harness exit non-zero by design, so this silently zeroed real
   successes.
5. **Token accounting was dead.** Tinker's SSE reply carries no `content-type`, so the
   proxy buffered every stream and lost its usage frames. Streaming is now detected from
   the request body, and usage is requested explicitly.
6. **ConTree grading was OOM-killed.** Sandboxes report 4 CPUs but only 4 GB while the task
   metadata declares 16 GB; the exported verifier builds with `cmake --build --parallel`
   and ignores `CMAKE_BUILD_PARALLEL_LEVEL`. Pinning commands with `taskset -c 0` makes
   `nproc` report 1 and fixes it. This alone moved the dev subset from 3/12 to 7/12.
7. **Session-title generation spent one request per episode** — the reference makes no such
   call. The profile initially disabled the wrong entry id
   (`session-title-first-prompt-llm` is the emitter).

## Reproducing

ConTree path (no cluster required):

```bash
python -m notes.dsh_inkling.harness.contree.run_batch \
  --tasks-file /tmp/dsh-contree/full-tasks.txt \
  --run-dir /tmp/dsh-contree/full02 \
  --concurrency 12 --cpu-affinity 0
```

Then audit:

```bash
python -m notes.dsh_inkling.analyze_run /tmp/dsh-contree/full02
```

The Rivanna path and its environment facts are in
[`harness/rivanna/RUNBOOK.md`](harness/rivanna/RUNBOOK.md).

## Status

The objective — outperform 52/100 — is met numerically but not statistically. The next
round should target the capped-episode population, where 62 episodes hold 32.3% pass rate
and the finished population already reaches 86.8%.
