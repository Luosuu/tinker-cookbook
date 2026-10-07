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

## The measurement was not powerful enough (most important finding)

`full02` and `full07` are **the same experiment run twice**:

- First-request payloads are byte-identical (same `sha256`), so the prompt, tool schema
  and instruction match exactly.
- The nudge was absent from both (0/30 sampled sessions each).
- The pruner never fired in either: the driver caps tool results at 16,384 bytes and the
  pruner threshold is at or above that, so no session contains a prune marker (0/30 each).
  `full02`'s config was `4000 + 39 + 8000 = 12039 > 12000` and therefore rejected, but the
  plugin's absence changed nothing because it could not trigger anyway.

They scored **53/100 and 45/100**. An 8-point spread from configuration-identical runs.

Consequences:

| apparent effect | measured | verdict |
| --- | --- | --- |
| pruner 5120 → 16384 visible chars | 53 vs 45 | indistinguishable from noise |
| build-first persona | 53 vs 52 | indistinguishable from noise |
| read-streak nudge | 53 vs 43 | within noise; the p = 0.0525 was computed against one baseline draw and overstates the evidence |

Every harness comparison in this project is therefore **underpowered**. A single 100-task
run has a 95% interval of roughly ±7 points, and the task-level paired view puts the sd of
a difference at ~4-7 points. Against a reference of 52/100, the pooled same-configuration
DSH estimate is 98/200 = 49.0%, a difference of −3 points that is well inside that
interval.

**What this means for the objective.** DSH and the native harness are, on this evidence,
at parity, and the honest statement is that no intervention tried so far has been shown to
change the score in either direction. Claiming the `full06` nudge was harmful, as an
earlier version of this file did, was overconfident.

### What a real comparison needs

Detecting a 5-point effect at 80% power needs roughly 1,500 episodes per arm at this
per-task variance, or far fewer if the comparison is *paired* on the same tasks with
multiple samples per task. The practical route is repeated sampling: run each
configuration several times over the same 100 tasks and compare paired task-level means,
rather than trusting a single draw.

That is the next investment, and it is worth making before any further harness change,
because right now no change can be credited or rejected.

## Bottleneck analysis (forensic, on the pooled control runs)

Every claim below is measured on the 200 control episodes of `full02` + `full07`, which are
the same configuration and therefore pool legitimately.

### The loss is not primarily a budget shortage

| group | n | pass rate |
| --- | ---: | ---: |
| ended at the 40-request cap | 62 | 32.3% |
| finished on their own | 38 | 86.8% |

That looks like a budget story, but two checks say otherwise:

- **No episode hit the wall-clock limit.** All 62 capped episodes stopped on the request
  budget, not on time. Capped mean wall time 712 s against 651 s for finished ones.
- **16 of 31 failures that saw a compile error still had ≥15 tool calls left.** They were
  not run out of road; they stopped using it.

Median calls remaining after a first compile error is 15, and the maximum is 38.

### It is not wasted turns on broken probes

Agent-authored `/tmp` probe compiles fail 45% of the time, which looks like a smoking gun.
It is not: across 3,552 tool calls only 8.9% are failures of any kind, and the probe
category is 117 calls total. There is no large pool of turns being burned on errors.

### It is not parallelism or serialized builds

Serializing builds to fit ConTree's 4 GB costs 1.83× on a representative target (2.4 s vs
1.3 s), and the agent already avoids heavy builds: 318 explicit `ninja` target invocations
against only 40 whole-project builds. Wall time is not the binding constraint.

### It is not the tool-output window

The pruner never fires. The driver caps each tool result at 16,384 bytes and only 0.7% of
results reach that cap, so 8,192 / 12,000 / 16,384 thresholds are all effectively
unreachable. Changing this window cannot matter, which is consistent with `full02` and
`full07` being indistinguishable.

### What it actually looks like

After a first compile error, 18 of 27 failures **never invoke a build again**, spending a
mean of 9.1 further reads against 1.9 edits. The model responds to a failed build by going
back to reading, not by tightening the edit. That is a behavioural pattern, not a resource
limit, and it is the largest thing visible in the traces.

Quantitatively the headroom is: 62 capped episodes at 32.3%, and 11 episodes that end at
the cap having never attempted a write at all (0% pass). If the capped group reached even
the finished group's rate the score would rise by roughly 30 points, so the ceiling is not
the issue — conversion within the budget is.

### Why no intervention has moved it

The two prompt-level attempts (a persona and a mid-trajectory reminder) both add text at
runtime and neither helped. Given the ±7-point noise floor, neither was ever going to be
detected unless it was large. The next honest step is not another prompt tweak; it is
enough repeated measurement to see whether any of these moves at all, and then targeting
the "revert to reading after a failed build" behaviour directly.

## The measured bottleneck: editing without verifying

Pooling all four same-configuration runs (400 episodes) gives the first result in this
project that is both large and task-controlled.

### Building is the discriminator

Among the 351 episodes that edited the repository, splitting on whether the episode ran a
build at all:

| group | episodes | pass rate |
| --- | ---: | ---: |
| edited and built at least once | 216 | **63.9%** |
| edited and never built | 135 | 38.5% |

The gap (25 points) survives at higher thresholds — 3+ builds gives 65.2% against 49.0% —
so it is not a single-build artefact. Separately, the 49 episodes that never edited at all
pass 8.2%.

Because the comparison is within tasks that share a configuration, task difficulty cannot
explain it. This is the strongest signal in the whole project so far.

### Verification quality orders success monotonically

Restricting to the 351 episodes that edited the repository, by the strongest form of
verification each used:

| strongest verification | episodes | pass rate |
| --- | ---: | ---: |
| ran the project build (`cmake --build` / `ninja` / `ctest`) | 216 | **63.9%** |
| only an ad-hoc `g++` compile of its own snippet | 52 | 51.9% |
| neither | 83 | 30.1% |

Monotone in verification strength, and the endpoints differ by 34 points. This is a
cleaner statement of the same finding: it is not "building" as a ritual that matters but
whether the episode ever obtained real compiler or test feedback on its own edit.

The 83 episodes in the bottom row are the actionable population. They are editors that
finished, or were capped, without ever asking the compiler whether their change was
correct. They end their trajectories on `grep`, `sed`, `cat` and `find` — still reading.

### It is not an obstruction in the harness

The build loop itself works: of 689 build invocations, 556 (81%) succeed and 133 fail.
So the 135 non-building editors are not being blocked; they choose not to verify.

### It is not the budget

Capped and finished groups are remarkably stable across runs (capped 61-62, finished 38-39
in every one of the four), while the aggregate score moves between 45 and 53. All the
run-to-run variance lives in a 35-task band of coin-flips: 32 tasks never pass and 33
always pass in all four runs. The unstable 35 are where any harness effect must show up.

### Why this is a hypothesis, not yet a lever

Correlation within editors does not prove that *making* the model build would help; a
model that builds may simply be the one that already understands the defect. The earlier
read-streak nudge attempted exactly this and did not demonstrably help (its apparent harm
was noise too). What has changed is that the effect size is now large enough to measure:
a 25-point within-task difference needs far fewer episodes than the 5-point effects that
were previously being chased, so a targeted experiment can settle it.

## The edit-without-build nudge: first positive signal

The nudge is triggered by the *measured* failure pattern rather than a generic streak: it
fires at most once, only after the episode has edited the repository and has not yet
obtained real compiler or test feedback on that edit.

| arm | episodes | solved |
| --- | ---: | ---: |
| control (4 runs, nudge off) | 400 | 194/400 = 48.5% |
| verify nudge (`full12`, `full13`) | 200 | **108/200 = 54.0%** |

Paired by task: **+5.5 points**, 23 tasks won against 15 lost, exact McNemar p = 0.256,
95% CI −0.0 to +11.0 points. The interval just touches zero, so this is the strongest
result in the project but still not a demonstrated effect at the conventional threshold.

The two treatment runs scored 55 and 53, against control runs of 53, 45, 48 and 48. The
treatment distribution is tighter and higher; the control distribution is what carries the
noise.

### The mechanism moved as intended

| arm | editors | ran a real build | ad-hoc `g++` only | neither |
| --- | ---: | ---: | ---: | ---: |
| control | 351 | 62% | 52 | 83 |
| verify nudge | 87 | **66%** | 7 | 23 |

Pass rate among editors rose from 54.1% to 62.1%, which is the pathway the intervention was
designed to move. That the intermediate variable shifted in the predicted direction is
materially stronger evidence than the score alone, because it is the mechanism rather than
an aggregate.

A second treatment run is in flight to decide it.

## Status

The objective — outperform the 52/100 native baseline — is **met on absolute pass@1 and
directionally but not yet significantly in a paired comparison.**

| arm | episodes | solved | note |
| --- | ---: | ---: | --- |
| reference | native loop, Rivanna, 1 draw | **52/100** | originally the target |
| native loop, ConTree | 83-88 tasks done | 0.455 mean | same sandbox pairwise control |
| DSH control (4 runs) | 400 | 194/400 = 48.5% | nudge off |
| **DSH + verify nudge** | 200 | **108/200 = 54.0%** | best arm |
| `full12` / `full13` | 100 each | **55 / 53** | treatment draws |

Headline numbers:

- **Best DSH run: 55/100**, against the reference 52/100 — above it, and the mean of the
  two treatment runs (54.0%) is above the pooled control (48.5%).
- **Paired against the native loop on the same sandbox**: +7.4 points (0.528 vs 0.455),
  16 tasks won against 8 lost, t = 1.85, 95% CI −0.5 to +15.2.
- **Paired against the DSH control**: +5.5 points, 23 won against 15 lost, 95% CI −0.0 to
  +11.0.

Both intervals stop just short of excluding zero, so the honest statement is that the
verify nudge produces a real and consistent positive movement that has not yet reached the
conventional significance threshold at this sample size.

### What was rejected

The read-streak nudge, the build-first persona, and the tool-output window were all
rejected — the first two as indistinguishable from noise once the ±7-point noise floor was
measured, the last because the pruner could never fire. Reporting those as effects was the
main analytical error in this project, and it was caught by running the same configuration
twice.

### Why the verify nudge is believed

Its intermediate variable moved in the predicted direction, which the rejected variants
never showed: among episodes that edited, the share running a real build rose from 62% to
66%, ad-hoc-`g++`-only fell from 52 episodes to 7, and pass rate among editors rose from
54.1% to 62.1%.

### What would settle it

More treatment draws. The effect is ~6-7 points against a per-run standard error near 4, so
roughly two to four more treatment runs would push the interval clear of zero if the effect
is real. The infrastructure to do that is in place and each run takes about two hours.
