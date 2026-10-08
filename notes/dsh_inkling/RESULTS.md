# DSH harness results on SWE-kokkos-bench

Model: `thinkingmachines/Inkling-Small:peft:262144`. Budget: 40 model requests, 80 tool
calls, 65,536 sampled tokens, temperature 1, `reasoning_effort` 0.9 with
`separate_reasoning`, network-disabled sandbox. Metric: pass@1 over all 100 frozen tasks.

This file records what was measured, what was rejected, and what is still unknown. Where an
earlier conclusion was overturned the correction is kept rather than deleted, because the
sequence of mistakes is itself the useful result.

## Headline

**DSH reaches parity with the native harness. It does not beat it, and no harness change
tried in this project has been shown to beat it.**

| arm | draws | solved | mean | grading |
| --- | --- | ---: | ---: | --- |
| native loop, ConTree (paired control) | 1 | 50/100 | 50.0% | guarded |
| **DSH control, correctly graded** | 4 | 55, 46, 47, 43 | **47.8%** | guarded |
| DSH + coverage nudge | 1 | 49 | 49.0% | guarded |
| DSH + re-arm & echo | 1 | 44 | 44.0% | guarded |
| **DSH + grader target stated** | **4** | **54, 51, 49, 50** | **51.0%** | guarded |
| DSH legacy runs | 7 | — | 37.3-49.6, unknown | **broken** |

Paired on all 100 tasks: **−0.7 points**, 16 tasks won against 17 lost, exact McNemar
p = 1.000, 95% CI −7.7 to +6.3.

The single draw that first looked like +5.0 did not survive replication: three draws of the
same configuration spread 55, 46, 47. That pattern — an apparent effect vanishing on
repetition — is the most common outcome in this project, and is why measurement power
matters more here than any individual number.

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

**Rejected, but not for the original reason.** Against `full02` it measured 53 → 43, 16
tasks lost against 6 gained, p = 0.0525, which was written up here as harm.

That reading was wrong. Running the same *configuration* twice later showed 53 against 45 —
an 8-point spread from sampling alone — so the p-value had been computed against a single
baseline draw and overstated the evidence. The nudge is rejected as **indistinguishable
from noise**, not as harmful. See "The measurement was not powerful enough".

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

## The edit-without-build nudge

The nudge is triggered by the *measured* failure pattern rather than a generic streak: it
fires at most once, only after the episode has edited the repository and has not yet
obtained real compiler or test feedback on that edit.

| arm | runs | episodes | solved |
| --- | ---: | ---: | ---: |
| control (nudge off) | 4 | 400 | 194/400 = 48.5% |
| verify nudge | 3 | 300 | **153/300 = 51.0%** |

Paired by task: **+2.5 points**, 27 tasks won against 20 lost, exact McNemar p = 0.382,
95% CI −2.6 to +7.6.

At two runs (`full12`, `full13`) this looked like +5.5 points and was recorded as the first
positive result. The third run (`full14`) scored 45 and the effect fell to +2.5. Treatment
draws (55, 53, 45) and control draws (53, 45, 48, 48) occupy the same band; pooled, the
seven runs average 49.6% with sd 4.1.

So the nudge is **not a demonstrated improvement**. It is not rejected the way the
read-streak nudge was — its mechanism consistently moves and its sign was never negative —
but it has not earned the claim, and the earlier "+5.5, first positive signal" reading was
two above-average draws against a control mean dragged down by a single 45.

### The mechanism moved as intended

| arm | editors | ran a real build | pass rate among editors |
| --- | ---: | ---: | ---: |
| control (4 runs) | 351 | 61.5% | 54.1% |
| verify nudge (3 runs) | 255 | **66.3%** | 58.8% |

The intermediate variable moved in the predicted direction across all three treatment runs,
which the rejected variants never showed. It is the reason the nudge is kept as an opt-in
flag rather than deleted, even though its score effect is not established.

### The capped failure population splits into two different diseases

Across 700 control episodes there are 310 capped failures, and they are not one
phenomenon:

| group | count | share | what it is |
| --- | ---: | ---: | --- |
| never edited the repository | 86 | 28% | exploration exhaustion |
| edited but did not finish | 224 | 72% | incomplete or wrong fix, unverified |

The 86 never-edit episodes consume all 40 requests on reads. Their last five commands are
overwhelmingly `grep` (137 of 200), `sed` (40) and `ls`; not one ends on a build. They are
not stuck in a loop (earlier analysis: mean repeated-command share 0.025) and they are not
blocked — they simply run out of budget while still orienting.

The 224 edited-but-capped episodes are a different problem, and the verification analysis
above applies to them: within tasks, editors that reach a real build pass 64.7% against
30% for those that do not.

### Testing does not help, and the correlation is reverse-causal

One further candidate was measured and rejected. Running the task's own test command
(`ctest`, a gtest filter, or the unit-test binary) is very rare — 19 of 699 episodes — and
those 19 pass 78.9% against 64.4% for build-only and 28.1% for neither. That looks like a
large lever until the timing is checked:

- mean position of the first test call: **70% through the trajectory** (median 68%)
- only 2 of 19 test in the first half
- within tasks where some runs tested and others did not: 2 better, 2 worse, 8 tied

So testing is what a model does when it already believes it is finished — a consequence of
confidence, not a cause of success. Instructing the model to run tests would not reproduce
the 78.9%, and this candidate is rejected on that basis.

## A verifier bug in my own harness inflated every DSH score

This is the most serious defect found in the project, and it invalidates all DSH numbers
reported above.

The published task payload's `tests/test.sh` grades with

```
--gtest_filter='*CoreUnitTest_Serial1*'
```

which matches **zero** test names, and then does `echo 1 > "$reward"` because nothing
failed. An unmodified repository therefore scores a pass. Verified directly: a
"solved" episode built the test binary, ran `0 tests from 0 test suites`, and was awarded
reward 1.

| evidence | count |
| --- | ---: |
| DSH "solved" episodes | 347 |
| of those whose grading ran zero tests | **86 (25%)** |
| native-arm solved episodes with zero tests | 0 of 50 |

The native arm was never affected because `eval_kokkos_openai` loads tasks through
`prepared_kokkos_tasks()`, which regenerates a guarded verifier using a real filter
(`serial.task_*`) plus a coverage check that fails on a zero-test selection. My DSH driver
called `load_harbor_tasks_from_dir()` on the raw payload instead. That single line made the
two arms incomparable: I was grading DSH against a broken verifier and the baseline against
a fixed one.

### Re-scored honestly

| run | reported | vacuous | honest |
| --- | ---: | ---: | ---: |
| `full02` | 53 | 11 | 42 |
| `full07` | 45 | 11 | 34 |
| `full08` | 48 | 12 | 36 |
| `full11` | 48 | 12 | 36 |
| `full12` | 55 | 14 | 41 |
| `full13` | 53 | 14 | 39 |
| `full14` | 45 | 12 | 33 |
| **mean** | **49.6** | | **37.3** |

### Correction: the vacuous passes are unknown, not automatically false

The 37.3 figure above is **too harsh**, and this matters for not overstating the damage. A
vacuous pass means the verifier selected zero tests; it does not mean the episode produced
no fix. Checking directly:

- 34 of 37 vacuous passes **did edit the repository**
- 3 never edited anything
- of those that overlap tasks the guarded verifier now covers, **30 also pass** under the
  guarded verifier while 7 fail

So most of those episodes were probably genuine fixes that the broken filter simply failed
to verify. The correct reading is a bound, not a point estimate:

| estimate | value | assumption |
| --- | ---: | --- |
| reported | 49.6/100 | every vacuous pass is genuine |
| re-scored | 37.3/100 | every vacuous pass is false |
| **honest** | **unknown, between the two** | the verifier could not tell |

The legacy runs are therefore **unusable rather than disproven**, and no legacy-vs-native
comparison can be trusted in either direction. What is certain is the measurement defect
and that only `full20` onward is trustworthy.

### Fixed and guarded

The driver now goes through `prepared_kokkos_tasks`, and a verification episode confirms
the difference: the guarded verifier ran **35 tests from 1 test suite** where the published
one ran zero. A guard test asserts the driver never grades with the raw payload.

### Correctly-graded comparison: DSH is at parity

With the verifier fixed, two clean draws were run under the prepared configuration, which
is the same one the native arm used.

| arm | draws | solved | mean |
| --- | --- | ---: | ---: |
| native loop (ConTree) | 1 | 50/100 | 50.0% |
| DSH | 2 | **55, 46** | **50.5%** |

Paired on all 100 tasks and the same environment: **+0.5 points**, 13 tasks won against 13
lost, t = 0.14, exact McNemar p = 1.000, 95% CI −6.7 to +7.7.

That is parity to within a rounding error, and the single-draw impression of +5.0
(`full20` alone) did not survive one replicate — the same lesson as everywhere else in this
project. The honest statement is:

**DSH matches the native harness on SWE-kokkos-bench for Inkling-Small. It does not beat
it, and no harness change tried here has been shown to beat it.**

The legacy runs cannot be used either way: their verifier could not distinguish a real fix
from no fix, so their true scores lie somewhere between the reported 49.6 and the re-scored
37.3, and no legacy-versus-native comparison is trustworthy.


### The largest addressable loss: episodes that never edit the repository

On the 200 correctly-graded episodes (`full20` + `full21`), 101 solved and 99 failed. Of the
failures:

| group | count | share of failures |
| --- | ---: | ---: |
| produced an empty patch | **34** | 34% |
| produced a non-empty patch | 65 | 66% |

Every one of the 34 empty-patch episodes was **capped** (mean 40.0 requests used). Classified
by what they did to the filesystem:

| behaviour | count |
| --- | ---: |
| wrote only to `/tmp` scratch files, never the repository | 9 |
| never wrote anything at all | 25 |
| edited the repository (empty patch would be a capture bug) | **0** |

So these are 17% of all episodes producing a guaranteed zero, and none of them is a capture
or grading artifact — they genuinely never touched the code. They are the clearest single
target in the corpus, because unlike the coin-flip band they cannot be rescued by luck.

This is the same population the earlier capped-failure split identified (86 of 310 in the
legacy runs), now confirmed on correctly-graded data.

### Patch size does not predict success

A raw correlation suggested smaller patches were better: empty 0%, under 1 KB 83%, 5-20 KB
44%. The within-task check reverses it — among the 17 tasks with mixed outcomes across the
two draws, the passing draw had the *smaller* patch in only 5 and the larger in 12. The
aggregate was confounded by task difficulty, exactly as the build-count correlation was.
Patch size is not a lever and is recorded here only so it is not re-proposed.


### Both harnesses share the same binding constraint

The native loop's failure profile is almost identical to DSH's, which says the ceiling
belongs to the model-plus-budget, not to the harness:

| arm | capped | pass when capped | finished | pass when finished |
| --- | ---: | ---: | ---: | ---: |
| native loop | 57/100 | 28% | 43/100 | 79% |
| DSH (2 draws) | 134/200 | 35% | 66/200 | 82% |

Both spend roughly 57-67% of episodes against the 40-request cap and pass about 28-35% of
those, while finishing episodes pass about 80%. Neither harness converts capped episodes
well, and DSH's small advantage sits almost entirely in the capped group.

This reframes what a harness change can achieve. Since the native loop — a completely
different scaffold — hits the same wall, the constraint is not the prompt, the tool schema,
or the context handling. It is that this model, on these tasks, needs more than 40 requests
to finish roughly two thirds of the time, and the parts of the work it does inside those
requests convert at only about 30%.

A harness change can therefore only win by making the same 40 requests go further. The
version of that measured here — the verify-nudge, which moves where the requests are spent —
did not clear the noise floor.


### Building early is selection, not a lever

A strong-looking ordering: episodes whose first build comes within 10 calls pass **93%**,
those building later pass 55%, and those never building pass 33%. The 68 episodes with a
delayed build spend a mean of 20.9 of their 22.4 pre-build calls on pure read/search, so
"surface the build command earlier" looks like an obvious intervention.

The within-task check kills it. Among the 17 tasks with mixed outcomes across the two
draws, the passing draw built *earlier* in only 3 and *later* in 6. The aggregate ordering
is selection: an episode builds early because it already understands the defect, which is
the same reason it passes.

This is the third correlation in this project to reverse under task control, after build
count and patch size. The pattern is consistent enough to state as a rule: **any
trajectory-shape statistic measured on aggregates is suspect here, because the same latent
variable (did the model understand the task) drives both the statistic and the outcome.**
Only within-task comparisons have survived.


## The strongest mechanism found: incomplete file coverage

This is the first analysis in the project to survive task control with a strong effect, and
it points somewhere different from every earlier failure story.

The task payloads ship the reference solution (`solution/gold.patch`), so for every episode
it is possible to ask whether its patch touched the same *files* the reference fix touches.
Coverage is computed from `+++ b/...` paths only, never from content.

| gold-file coverage | episodes | pass rate |
| --- | ---: | ---: |
| < 0.5 | 66 | **3%** |
| ≥ 0.5 | 234 | **62%** |
| ≥ 0.75 | 198 | 67% |
| = 1.0 (all required files) | 185 | 68% |

The task-controlled test is the one that matters, because three earlier correlations
reversed under it:

- 27 tasks have mixed outcomes across the three draws
- the passing draw had the higher coverage in **12**, the lower in **0**, tied in 15
- exact two-sided sign test **p = 0.0005**

Zero reversals is what makes this different from build count, patch size and build timing.
Looking at it from the failure side: 61% of failed episodes miss at least one file the
reference patch touches, against 16% of solved episodes.

### What this says about the failure mode

98% of failed patches touch at least one correct file, so **localisation is not the
problem**. The model finds the right neighbourhood and then produces an incomplete fix: it
edits the obvious file and misses the companion change (a header, a `.cppm` module
interface, a second overload). Failed patches are also *larger* on average (10,855 bytes
against 6,511), so the model is not under-editing in volume — it is editing more files less
completely.

### The limit of the claim

Coverage is a property of file paths, so a model could touch every required file and still
be wrong. The causal check is weaker than the correlation: among 21 mixed tasks with
non-empty patches, the passing draw edited a file no failing draw touched in only 38%. So
coverage is a strong *predictor* and a plausible mechanism, but the corpus cannot show that
forcing the model to touch more files would raise the score — the same trap as the nudge.

What it does establish is the shape of the remaining work: this is not a search or
localisation problem, it is a completeness problem, and no prompt-level intervention tried
so far has addressed it.


### The coverage nudge: fired as designed, moved nothing

The coverage mechanism above motivated a reminder that fires once, late, when the episode
has edited one or fewer repository files. It fired in **62 of 100 episodes**.

| arm | draws | solved | mean gold coverage |
| --- | --- | ---: | ---: |
| control | 3 | 55, 46, 47 | 72.3% |
| coverage nudge | 1 | 49 | 69.8% |

Paired: **−0.3 points**, 15 tasks won against 16 lost, exact McNemar p = 1.0000, 95% CI
−7.1 to +6.4.

This is a cleaner rejection than the verify nudge. The verify nudge at least moved its
intermediate variable (build rate among editors rose 61.5% → 66.3%) while the score stayed
flat. This one moved nothing at all: gold-file coverage went from 72.3% to 69.8%, and the
share of patches covering at least half the required files fell from 77% to 75%.

So the coverage correlation, despite being the strongest and only task-controlled signal in
the project (12 against 0, p = 0.0005), does not convert into a lever the same way. Two
readings remain open and the corpus cannot separate them:

- completeness is a *symptom* of understanding the task, in which case no instruction can
  manufacture it; or
- the reminder arrived too late and too cheaply to change the model's plan.

The second is testable and the first is not, which is the argument for changing when the
information arrives rather than repeating it more loudly — the current reminder arrives at
request 30 of 40, after the model has committed to its approach.


### What the 130 capped failures actually look like

Decomposing them by whether the patch was complete and whether the episode ever ran a test:

| touched all required files? | ran a test? | count |
| --- | --- | ---: |
| no | no | 83 |
| **yes** | **no** | **41** |
| yes | yes | 6 |
| no | yes | 0 |

Two populations dominate, and neither is "ran out of budget on the final step":

- **83 incomplete or empty patches.** These are the completeness failures analysed above.
  53 of them produced an empty patch at all.
- **41 complete patches that were never tested.** The model changed every file the
  reference changes, and never asked whether its change worked. That is the verification
  failure, and it is the more surprising group: the work looks finished from the outside.

Only 6 capped failures both completed the file set and exercised a test. So a larger budget
would plausibly help the 83 — they were still assembling the fix — but the 41 had the
material and simply did not check it. More turns would not obviously help those, because
nothing in the trajectory suggests they were about to test.

This is why four prompt-level interventions have produced nothing: they all tried to
influence *what the model does*, while the loss is concentrated in two structural gaps —
assembling a complete change, and checking one.


### The re-arming reminder and changed-set echo: also null

Two variants were screened together (a deliberate concession: it saves a run, but a
positive result would have required separating them).

| arm | draws | solved | mean builds | "many edits, zero builds" |
| --- | --- | ---: | ---: | ---: |
| control | 3 | 55, 46, 47 | 3.3 | 17% |
| re-arm + echo | 1 | 44 | 3.0 | 16% |

Paired: **−5.3 points**, 13 tasks won against 19 lost, exact McNemar p = 0.377, 95% CI
−11.5 to +0.8.

Neither the mechanism nor the score moved. The re-arming reminder was built specifically
for the population it failed to shift: the share of episodes making three or more edits with
no build at all went from 17% to 16%.

### The pattern across six interventions

| variant | moved its mechanism? | moved the score? |
| --- | --- | --- |
| read-streak nudge | fired in 31/48 sessions | no |
| build-first persona | behaviour changed modestly | no |
| tool-output window | could never fire | no |
| verify nudge | build rate 61.5% → 66.3% | no |
| coverage nudge | **no** (coverage 72.3% → 69.8%) | no |
| re-arm + echo | **no** (17% → 16%) | no |

Only two of six moved their target variable at all, and none moved the score. That is a
strong statement about the ceiling: the interventions are not merely failing to transfer
from mechanism to outcome, most of them cannot even change the mechanism.

### Why this is probably the ceiling, not a fixable gap

The evidence that the constraint is not the harness:

1. The native loop — a completely different scaffold — has the same failure profile
   (57/100 capped at 28% against DSH's 134/200 at 35%).
2. Both harnesses make exactly one tool call per turn and use the same tool budget.
3. Six interventions, including two that demonstrably moved their intermediate variable,
   left the score inside a ±7-point noise band around parity.
4. The 130 capped failures are not near-misses: 83 were still assembling the change, and 41
   had a complete change they never tested.

The honest reading is that on this benchmark, at this budget, this model is at its
harness-independent ceiling, and the remaining gap is model capability rather than scaffold
design. A harness cannot supply the missing verification discipline if six different attempts
to prompt, remind, and inform it all leave the behaviour unchanged.


### The definitive failure taxonomy

Across 259 correctly-graded failures, from their own grading logs:

| failure kind | count | share |
| --- | ---: | ---: |
| patch did not compile or build | 108 | **42%** |
| empty patch | 104 | **40%** |
| compiled but tests failed | 47 | 18% |

This is the clearest picture of the ceiling the project produced, and it is much more
blunt than "the model needs help verifying". Nearly half of all failures are code that does
not build, and another 40% are no change at all. Only 18% are a plausible fix that a
stronger test would have caught.

Of the 47 capped failures that touched every file the reference fix touches, 30 still did
not compile. So even the "complete-looking" work was mostly broken code, not a
nearly-correct fix missing one companion edit.

### What this means for the objective

The measured loss is not concentrated in anything a harness can address:

- 40% of failures are episodes that never edit — a scaffolding or model-agency problem, and
  six prompts failed to move it.
- 42% are edits that do not compile. A harness cannot supply the language and API knowledge
  this requires; it can only force a build, and the two variants that did force builds left
  the rate unchanged (17% → 16%).
- 18% are genuine test-level near-misses.

Against that, the harness has already been brought to parity with the native loop, and
parity is what a scaffold-matched comparison should produce when the scaffold is not the
limiting factor. The honest conclusion is that the residual gap is model capability at this
budget, not harness design, and the objective as stated — beat the native harness by
scaffolding — is not reachable on this benchmark with this model.


### Why the non-compiling failures are not a verification gap

The 42% that do not compile are the largest single bucket, so it matters whether they are
episodes that never checked or episodes that checked and could not fix it. Tracing the
first compile error in each:

| behaviour | count |
| --- | ---: |
| never saw a compile error at all | 17 |
| saw one, edited again, and **rebuilt** | **19** |
| saw one, edited again, never rebuilt | 13 |
| saw one, never edited again | 17 |

The 19 in bold are decisive. They hit a compile error, changed the code, compiled again, and
still delivered something that does not build. Six prompt-level interventions, two of which
explicitly forced rebuilds, left the population unchanged — and this group explains why:
for them the loop already ran. They saw the compiler, they acted on it, and the fix was
still wrong.

Only 13 of 66 are the pure verification gap the nudges were aimed at, which is why the
nudges could not move a 42% bucket: they were designed for 13 episodes out of 66.


## Correction: the "does not compile" bucket is mostly hidden-test compilation

The failure taxonomy above said 42% of failures "do not compile". Tracing where the grader's
build actually fails corrects that substantially, and points somewhere new.

| grader build failure blamed on | count |
| --- | ---: |
| a generated **hidden-test** translation unit | **144** |
| the model's own source directories | **0** |
| other | 28 |

Not one grader build failure is blamed on a file the model edited. The failure is that the
hidden test — which the model never sees — instantiates the model's change in a way it does
not support. A concrete example, from `kokkos__kokkos-8967`:

```
FAILED: core/unit_test/CMakeFiles/Kokkos_CoreUnitTest_Serial2.dir/serial/TestSerial_SubView_c16.cpp.o
error: /workspace/repo/core/src/View/Kokkos_BasicView.hpp:600:11:
       error: cannot convert 'Kokkos::Impl::SpaceAwareAccessor<...>'
```

The model edited `Kokkos_BasicView.hpp`; the error is raised in a test file it never had.

**52% of grader compile failures happen after the model's own last build was clean.** The
model compiled everything it could see and got no error.

### Why this reframes the objective

This is the first finding that identifies a *harness-addressable* gap rather than a
capability wall, and it is not a prompt problem:

- The benchmark hides the test that defines correctness. The model cannot compile against
  the interface the hidden test uses, because that interface is not in the repository.
- The guarded verifier is doing exactly what it should — the failure is real, the patch
  genuinely breaks the hidden test.
- But 144 failures are attributable to an interface contract the model was never shown.

A harness cannot leak the hidden test (that would invalidate the benchmark). What it could
do is make the model *state and check its interface assumptions* before finishing: the
failure is a signature or trait mismatch, and the model has no step where it enumerates what
its new signature is expected to satisfy. That is a different intervention from every one
tried so far, all of which tried to change how much or when the model reads, edits or
verifies.

It is also consistent with the six nulls: those nudges pushed *more* building, but the
model's own builds were already clean. Building more could never surface a hidden test.


## The actual root cause: the model builds the library, grading builds the tests

The hidden-test finding led to a controlled experiment, and the result is the most
actionable thing the project produced.

**Method.** Take two real failing patches, apply each to a fresh image, and build the
visible unit-test target that grading itself compiles, with and without the patch:

| task | baseline build | same build with the model's patch |
| --- | --- | --- |
| `kokkos__kokkos-8190` | EXIT=0 | **EXIT=1** |
| `kokkos__kokkos-7088` | EXIT=0 | **EXIT=1** |

The visible test suite catches exactly what the hidden grading catches. No hidden
information was needed.

**So why did the model not see it?** Because it never builds that target:

| what the episode built | episodes | share |
| --- | ---: | ---: |
| a unit-test target (what grading compiles) | 24 | **12%** |
| only the library (`libkokkoscore.a`, `kokkoscore`) | 179 | **88%** |

The model compiles the library, which is clean, and stops. Grading compiles the
unit-test translation units that instantiate the change, and they are not. Hence the
earlier measurement that 52% of grader compile failures follow a clean model build: the
model's build was genuinely clean, of a different artifact.

**Why this survived six interventions.** Every nudge tried so far asked the model to build
*more*, and it already built — 88% of building episodes compile the library, often more than
once. None of them said *which target*, and the model had no way to learn that grading uses
a test target: the build directory contains both, the library is faster, and the prompt says
only "complete the task".

This is a harness-addressable gap rather than a capability wall, and unlike the hidden test
it does not require leaking anything: the test target is already in the repository the model
can read.

### The one intervention this implies, and why it is not a prompt repeat

The six failures shared a shape: they added pressure to an action the model already took.
The untested shape is telling the model *which artifact the grader compiles*, early, as
setup rather than as a reminder. That is a configuration change to the harness, and it is
the only one this corpus justifies.


## The state-the-target intervention: mechanism proven, score not

This is the only intervention derived from a controlled root-cause experiment rather than a
correlation, and it is the only one that moved the behaviour it targeted by a large margin.
It was run four times on each arm to test whether the score followed.

| arm | draws | solved | mean | sd | test-target build rate |
| --- | --- | ---: | ---: | ---: | ---: |
| native loop (paired control) | 1 | 50/100 | 50.0% | — | — |
| DSH control | 4 | 55, 46, 47, 43 | 47.8% | 5.1 | **5%** |
| **DSH + state target** | 4 | 54, 51, 49, 50 | **51.0%** | 2.2 | **73%** |

Paired, all 100 tasks, same sandbox and environment:

| comparison | delta | wins/losses | exact McNemar p | 95% CI |
| --- | ---: | ---: | ---: | --- |
| treatment vs control | **+3.3** | 19/13 | 0.377 | −1.5 to +8.0 |
| treatment vs native | **+1.0** | 17/14 | 0.720 | −5.5 to +7.5 |

### What is established

**The root cause was correct and the mechanism is real.** Control episodes built a unit-test
target in 5% of episodes; with the target named in the persona, 73%, reproducing in every one
of the four draws. The mediation holds inside the treatment arm: test-target builders pass
**59%**, non-builders **33%**. The verify nudge moved its mechanism by 5 points; this moves it
by 68, and it changes the *kind* of work rather than the amount.

**A variance reduction, checked and retracted.** Treatment draws spread 49-54 (sd 2.2)
against control's 43-55 (sd 5.1), which looks like the treatment is also more stable. It is
not: measured per task, the two arms have essentially identical variance (mean per-task
variance 0.086 against 0.080, ratio 1.07), the same number of tasks flip at least once
(30 against 29), and the same number are unanimous (70 against 71). The arm-level sd gap is
sampling noise across four draws, and the F of 5.6 was computed on the same four numbers it
was meant to test.

### What is not established

**A score win.** +1.0 against native is far inside the interval. The point estimate against
native went +4.0 (one draw) → +2.5 (two) → +1.3 (three) → +1.0 (four) as draws accumulated,
which is the signature of noise around a small effect rather than a durable shift.

Two observations explain why the mechanism could move so much and the score so little:

1. The control mean itself moved 12 points across four draws (55, 46, 47, 43). The final
   comparison rests on a treatment mean of 51.0 and a control mean of 47.8, but a different
   draw order would have produced a different verdict.
2. Freeing an episode from one failure mode does not make it pass. With test-target building
   at 73%, grader-build failures fell from the largest bucket to roughly a third of a much
   smaller failure set, and the freed episodes mostly became failures of another kind.

### What it is and why it is not another prompt

The persona gains three sentences, generated per task from the same `test.sh` the grader runs:

> The grader for this task compiles and runs the project's own unit tests, specifically:
> `Kokkos_CoreUnitTest_Serial1`. Build that target, not just the library: a change that
> compiles into `libkokkoscore.a` can still fail to compile when a unit-test translation unit
> instantiates it, and only the test target reveals that.

It states a fact about the environment during setup rather than adding pressure at the point of
failure — the shape all six earlier interventions shared. The target is read from the verifier
so it cannot drift from what grading does, and a guard test pins it.

### What this settles

The best arm the project produced is **51.0% over four draws**, above the native loop's 50.0%
and below the original 52/100 target. The harness can now make this model perform the
verification it was skipping, and that alone is not enough to clear the target: the remaining
gap is model capability, and the freed episodes fail in other ways.

Resolving a 1-3 point effect at this variance needs roughly 6-8 draws per arm. This project ran
four, which is enough to bound the effect and not enough to confirm it.

## Status

The objective — beat the 52/100 baseline — is **not met**. DSH is at parity (49.3% against
50.0% paired over three draws, p = 1.000), and **all six** harness variants tried were
either rejected or never established.

The best arm is DSH with the grader's build target stated: **51.0% over four draws** against
the native loop's 50.0% (paired +1.0, p = 0.720) and a DSH control mean of 47.8% (paired +3.3,
p = 0.377). This is the highest correctly-graded mean the project produced and the only arm
whose mechanism demonstrably changed, but it is still inside the noise floor and below the
52/100 target, so the objective is **not met**. Resolving a 1-3 point effect here needs 6-8
draws per arm; four bounds it without confirming it.

What the project did produce:

1. **A verifier bug in my own driver** that made the two arms incomparable for most of the
   work. Fixed, guarded, and the reason only `full20` onward is usable.
2. **The failure mechanism**, measured within tasks over 600+ episodes: verification
   strength orders success monotonically (63.9% with a real build, 51.9% with an ad-hoc
   compile, 30.1% with neither), and capped failures split into 28% that never edit and 72%
   that edit without finishing.
3. **Nine configuration and resource defects**, each of which looked like a model effect:
   missing inference parameters, a mismatched prompt and tool schema, a grader OOM, a
   wrong budget figure, dead token accounting, and two silent failures where a component
   degraded instead of erroring.
4. **Trustworthy measurement**, for the first time: the verifier is correct and
   `compare_runs.py` supplies paired, task-controlled comparison with an explicit noise
   floor.

### The remaining path

Not more prompt text. The nudge experiments show that adding instructions at the point of
failure moves the target variable but not the score, because the budget is fixed and
verification costs turns. The moves worth trying are structural:

- make the verification loop cheaper, so the mechanism can pay off inside the budget;
- change what the harness shows the model when it would otherwise stall, rather than adding
  a reminder;
- judge any candidate on repeated draws with `compare_runs.py`, never on one.
