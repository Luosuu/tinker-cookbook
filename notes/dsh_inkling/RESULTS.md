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

## Read-streak nudge

A bridge-level nudge appends a short reminder to the tool result after 12 consecutive
commands that neither edit nor build. It is free: it rides on a result the model already
receives, so it cannot consume the request budget. `HPC_NUDGE_AFTER=0` disables it.

Verified to fire: 31 of 48 sessions received at least one nudge.

**It hurts, and is rejected.** Against `full02`, with the nudge the only changed variable:

| run | nudge | solved |
| --- | --- | ---: |
| `full02` | off | **53/100** |
| `full06` | on | 43/100 |

16 tasks lost against 6 gained, exact McNemar two-sided p = 0.0525. A ten-point drop with
a p-value at the conventional threshold is strong evidence the nudge is harmful, not
merely unhelpful.

The most plausible mechanism is that the reminder interrupts an otherwise productive
exploration plan: it fires on a *streak*, which cannot distinguish an agent that is stuck
from one that is reading a large header before making a single correct edit. Episodes are
also visibly longer under the nudge, so it consumes wall-clock without buying turns.

The correlation that motivated it — capped failures average 2.3 builds against 5.2 for
capped successes — was a property of outcomes, not a lever: pushing the model to build
more does not make it build *the right thing*.

## Two failures that invalidated whole runs

Both were silent, and both are now guarded by tests in
[`harness/contree/run_task_contree_test.py`](harness/contree/run_task_contree_test.py).

1. **A rejected plugin config does not stop the run.** The pruner rejects any config where
   `headChars + len(marker) + tailChars > thresholdChars`, and the marker is 39 code
   points. Both attempts were over budget, so the pruner failed to load in all 200
   episodes of `full02` and `full03` and the runs used the 8192 defaults while the docs
   claimed 16384.

2. **A half-applied bridge edit hung every episode.** The nudge was added in two pieces
   and only one landed, leaving `nudge` referenced but undefined. The tool callback threw,
   no result reached the harness, and every episode stalled forever after its first
   request with node idle in `kevent`. It presented as 12 episodes frozen at exactly one
   request, and bisecting against the previous bridge revision confirmed the cause.

The lesson both share: in this harness a broken component degrades quietly rather than
failing loudly, so a run's own logs must be checked before its score is believed.

## Harness variants tested

### Build-first persona

Motivated by the capped population: episodes that stop at the request cap and succeed
average 5.2 builds, while those that stop at the cap and fail average 2.3, so the failing
group keeps reading instead of compiling.

Controlled A/B on a 30-task stratified subset (15 tasks the reference solves, 15 it does
not, seed 20261005), same profile and budget, persona the only variable:

| arm | persona | solved |
| --- | --- | ---: |
| A | reference | 13/30 |
| B | build-first | **16/30** |

+3 tasks (+10 points), 4 gained against 1 lost. Exact McNemar two-sided **p = 0.375** —
suggestive, not significant. The persona moved behaviour in the intended direction
(3.2 → 3.5 builds per episode) but modestly.

**The full 100-task run did not replicate it.** Against `full02`, with the persona the only
changed variable:

| run | persona | solved |
| --- | --- | ---: |
| `full02` | reference | **53/100** |
| `full03` | build-first | 52/100 |

9 gained against 10 lost, exact McNemar p = 1.000. The 30-task subset result was noise:
with 13-16 of 30 solved, a ±3 swing sits well inside the interval. Recorded as a negative
result — the persona is not adopted and the +3 must not be cited as a win.

### Act-early persona

6/12 on the dev subset against 7/12 for the reference persona. No evidence of improvement;
the subset is too small to resolve anything, and this variant is not pursued.

### Rejected

- A repetition breaker: capped empty-patch episodes average a 0.025 repeated-command share
  with none above 50%, so there is no looping to break.

## Status

The objective — outperform 52/100 — is **met numerically but not statistically, and no
harness variant has improved on it.**

| run | harness | solved | note |
| --- | --- | ---: | --- |
| reference | native Chat Completions (Rivanna) | **52/100** | the target |
| `full02` | DSH, reference persona, pruner silently at defaults | **53/100** | tie: 23 discordant, p = 1.000 |
| `full03` | + build-first persona | 52/100 | wash: 9 gained / 10 lost, p = 1.000 |
| `full06` | + read-streak nudge | 43/100 | **worse**: 6 gained / 16 lost, p = 0.0525 |

Best DSH result is 53/100, one task above the reference and well inside noise. Two
variants were tested and both were rejected, one of them clearly harmful. The honest
reading is that DSH, once its prompt, inference parameters and budget are matched to the
reference, performs at parity with the native harness — which is itself a meaningful
result, because three separate configuration gaps initially made it look far worse.

### What is left

The loss is concentrated in episodes that exhaust the 40-request budget: 62 of 100 in
`full02`, passing 32.3%, against 86.8% for the 38 that finish on their own. Neither a
persona nudge nor a mid-trajectory reminder moved that; both are prompt-level interventions
and the model appears to be near its ceiling within this budget.

The untested levers are structural rather than prompt-level: giving the agent a cheaper
build/test loop (the build cost is what makes early editing expensive), or changing what
the harness shows the model at the point it would otherwise stall. Any such change must be
judged on the full 100 tasks, since a 30-task subset was already shown to produce a
spurious +3.
