# Handoff: DSH harness iteration for Inkling-Small on SWE-kokkos-bench

Branch `dsh-inkling-kokkos`, 79 commits from `2b49b47`. Written so a fresh agent can continue
without re-deriving the project. Read [`RESULTS.md`](RESULTS.md) for the full evidence chain;
this file is the orientation, the reusable lessons, and the open threads.

## Objective and status

Iterate the DSH harness until Inkling-Small beats the 52/100 native Chat Completions baseline
on SWE-kokkos-bench, at a matched budget. **Not met.** Two same-period blocks (all arms run
together, four draws each) put DSH + state-target at +3.8 and then +0.2 over native; combined
+2.0, p ≈ 0.10. Adding `--managed-checkpoint` gave +1.0 over native in the second block, not
significant. The 52/100 reference was one early native draw; same-period native draws average
48-50. The first-edit nudge was rejected as a time confound.

Run everything through the ConTree sandbox; no cluster needed. The Rivanna path is kept in
`harness/rivanna/` but SSH to its login node was blocked from this host.

## The numbers, and which ones to trust

**Correctly graded only.** Everything before `full20` is unusable; see defect 1 below.

| arm | draws | solved | mean | note |
| --- | --- | ---: | ---: | --- |
| native loop | 3 | 50, 50, 40 | 46.7% | also `native01` at concurrency 4 |
| DSH control | 6 | 55, 46, 47, 43, 43, 49 | 47.2% | |
| DSH + state-target | 7 | 54, 51, 49, 50, 50, 48, 46 | **49.7%** | best arm |
| DSH + coverage nudge | 1 | 49 | 49.0% | rejected |
| DSH + re-arm & echo | 1 | 44 | 44.0% | rejected |
| DSH + first-edit nudge | 3 | 56, 51, 52 | 53.0% | `full36`-`full38`; rejected: time confound |
| DSH control, later period | 1 | 52 | 52.0% | `full39`, concurrency 8, same session as `full38` |
| **same-period block** native | 4 | 45, 47, 49, 51 | 48.0% | `native04`-`07`; pass@4 63% |
| **same-period block** control | 4 | 45, 51, 48, 49 | 48.2% | `full40`-`43`; pass@4 60% |
| **same-period block** state-target | 4 | 52, 55, 50, 50 | **51.7%** | `full44`-`47`; pass@4 **67%** |
| block 2 native | 4 | 49, 53, 51, 46 | 49.7% | `native08`-`11`; pass@4 63% |
| block 2 state-target | 4 | 49, 49, 53, 49 | 50.0% | `full48`-`51`; pass@4 63% |
| block 2 state-target + checkpoint | 4 | 54, 50, 48, 51 | 50.7% | `full52`-`55`; pass@4 67% |
| (discarded) | — | 53, 45, 48, 48, 55, 53, 45, 33, 29, 32 | — | broken verifier or compromised |

**Parity between the unmodified harnesses is the firmest result here**: 47.2 against 46.7
across nine draws, a 0.5-point gap. Every harness change either failed or produced a small
effect this measurement cannot resolve.

## The lessons that cost the most

1. **Audit your own harness before believing any score.** The published task payload's
   `tests/test.sh` grades with `--gtest_filter='*CoreUnitTest_Serial1*'`, which matches zero
   tests, then awards reward 1 because nothing failed — an unmodified repo passed. 86 of 347
   DSH "solves" were graded that way. The native arm was immune because it loads tasks through
   `prepared_kokkos_tasks()`, which regenerates a guarded verifier. My driver called
   `load_harbor_tasks_from_dir()` on the raw payload. **One line made the two arms incomparable
   for most of the project.** Fixed and guarded by a test.

2. **The noise floor is ±7 points at n=1.** The same configuration ran 53 and 45; the control
   ranged 43-55; the treatment 46-54. Resolving a 2-4 point effect needs 6-8 draws per arm.
   Three early "positive results" were retracted for this reason, including one at p = 0.0525.

3. **Aggregate correlations reverse under task control.** Build count, patch size, and build
   timing all looked predictive in aggregate and all reversed or vanished within tasks, because
   one latent variable (did the model understand the task) drives both the statistic and the
   outcome. **Only within-task comparisons survived** — the coverage finding (12 against 0,
   p = 0.0005) and the state-target mechanism. Before proposing a lever from a correlation,
   check it within tasks.

4. **A moved mechanism does not imply a moved score.** Six interventions were null. Two moved
   their target variable; only one moved it by a lot (5% → 73% test-target building) and that
   still bought only ~3 points, because freed episodes failed a different way.

5. **Reminders cannot manufacture work the model does not do.** The coverage reminder fired on
   all 17 never-edit episodes of one run; none edited. The verify nudge fired in 62% of episodes
   and moved nothing. This is why the seventh intervention changed the *kind* of information
   (state a fact about the environment during setup) rather than adding pressure at the point of
   failure.

6. **Read a run's own logs before trusting its score.** Two silent defects: a rejected plugin
   config that fell back to defaults for 200 episodes, and a half-applied bridge edit that left
   `nudge` undefined and hung every episode after its first request. Audit every run for
   `error`-bearing zeros and `not_started` episodes — that is how the compromised control draw
   (`full32`, 32/100) was caught and excluded.

7. **Concurrency is a controlled variable with a settling time.** Two 8-way × 4-CPU batches
   launched together collapsed the within-group pass rate from 82% to 36-47% while capped
   counts, build health, and OOM checks all looked normal. Hold total vCPU fixed and check the
   within-group pass rate, not just the aggregate.

8. **4 CPUs per episode OOMs the verifier.** I measured 4 CPUs as 3.4× faster on an isolated
   target and changed the default; real runs then had 12 grading OOMs against 2, because the
   verifier runs its own `--parallel` build inside 4 GB. Reverted to 2 and guarded by a test.

## The failure taxonomy (259 correctly-graded failures)

| failure kind | count | share |
| --- | ---: | ---: |
| patch did not compile or build | 108 | 42% |
| empty patch (never edited) | 104 | 40% |
| compiled but tests failed | 47 | 18% |

Correction found late: of the grader build failures, **144 of 172 are blamed on a generated
hidden-test translation unit and 0 on the model's own source**. The model compiles the library,
which is clean, while grading compiles the unit-test target, which is not. That is the root
cause the state-target intervention addresses.

## The one intervention that works

`--state-target` appends three sentences to the persona, generated per task from the same
`test.sh` the grader runs, naming the target grading compiles and saying the library alone is
not enough. Mechanism: test-target build rate **5% → 73%**, reproducing in all seven draws,
mediating the outcome (builders pass 59%, non-builders 33%). Score: **+2.5 to +3.0 points**,
p = 0.272/0.659.

It is in `harness/contree/run_task_contree.py::_target_brief`, pinned by a guard test.

## Open threads, in priority order

1. **Run one more same-period block of state-target against native.** The first block
   (`harness/contree/launch_pass4.sh`: 12 runs × 3 episodes, 36 sandboxes, 4.7 h) gave +3.8
   pass@1, p ≈ 0.08 alone. A second block with only those two arms, six draws each at the
   same total concurrency, should take a similar time and would settle the objective on its
   own. Analyse with `compare_runs.py` (it reads native runs and prints pass@k) and compare
   only within a block; never pool an arm from one period against a baseline from another
   (that is how the first-edit nudge appeared to gain +5.8). Keep concurrency
   fixed within a draw: at 10, `full37` lost 80 of 100 tasks to ConTree/API failures. ConTree
   needs `full_network` from the agent sandbox; the earlier 403s were the sandbox proxy. A
   detector flaw remains: episodes that write only `/tmp` scratch files disable the reminder.

2. **Test the managed-checkpoint arm.** Across seven `--state-target` draws, only 160 of 559
   non-empty patches (28.6%) had a successful target build after their final edit. Piped builds
   hid 150 real failures behind exit 0, and stale/absent post-edit feedback covered 88 of 104
   final compile failures. The new `--managed-checkpoint` makes build-pipeline status truthful
   and runs at most two late incremental builds of the visible grader target. The next arm is
   `--state-target --managed-checkpoint`; its primary metric is truthful-fresh target coverage,
   not the first score.

3. **A systemic verifier timeout is unfixed** and costs ~3 points per run on both arms.
   `kokkos__kokkos-8182` times out in 10 of 11 runs: its verifier builds
   `Kokkos_ContainersUnitTest_Serial --parallel` and the episode ends at 1367-1410 s. Raising
   `grader_timeout` to 2400 s did **not** fix it — a direct re-run took 2686 s and still failed,
   because the sandbox's 3600 s lifetime is what binds. The default stays at 2400 s for
   merely-slow verifiers. The task is deliberately kept in the set; removing it would flatter
   every number.

4. **The 52/100 reference is a single sample.** The native arm's per-task disagreement between
   two draws is 18 of 100, identical to DSH's 19.4, so the original target carries the same ±7
   point interval as everything else.

5. **Untried levers.** The "compiled but tests failed" group has not been attacked directly.

## How to reproduce

```bash
# one draw (control); add --state-target for the working variant
python -m notes.dsh_inkling.harness.contree.run_batch \
  --tasks-file notes/dsh_inkling/harness/contree/tasklists/full100.txt \
  --run-dir /tmp/dsh-contree/fullNN --concurrency 10 --cpu-affinity 0,1 \
  --persona notes/dsh_inkling/harness/contree/persona-reference.txt

# native baseline
python notes/dsh_inkling/harness/native/run_native_contree.py \
  data/kokkos/SWE-kokkos-bench-v2 /tmp/dsh-contree/nativeNN \
  /tmp/dsh-contree/cache/images.json 10

# compare arms, paired by task
python -m notes.dsh_inkling.compare_runs \
  --arm control=/tmp/dsh-contree/full20,/tmp/dsh-contree/full21 \
  --arm treat=/tmp/dsh-contree/full25,/tmp/dsh-contree/full26
```

Notes: the bridge is copied into `.dsh-runtime/` before each run
(`cp harness/contree/sandbox-bridge.mjs .dsh-runtime/`). Runs take 2-3 hours at concurrency 10.
`pytest notes/dsh_inkling/` has 48 tests, including guards for the verifier bug, the CPU
default, and the target brief. Add `ruff check notes/dsh_inkling/` before committing.

The public-repo discipline in `AGENTS.md` applies: keep internal context out of commits.

## Available flags (all opt-in, all defaulting off)

| flag | what it does | verdict |
| --- | --- | --- |
| `--state-target` | names the target grading compiles | mechanism works; score +2.0 vs native over two blocks, p ≈ 0.10 |
| `--managed-checkpoint` | truthful piped-build status + up to two late target builds | fired in 29% of episodes, mechanism +2.4 points, score within noise |
| `--env-brief` | repo layout and build facts read from the checkout at setup | smoke-tested only |
| `--first-edit-nudge` | repeats the budget arithmetic while nothing is edited | rejected (time confound) |
| `--coverage-nudge` | fires late if only one file was edited | rejected |
| `--verify-rearm` + `--echo-changed` | re-arms the build reminder; echoes the changed set | rejected |
| `--verify-nudge` | one-shot build reminder | rejected |
| `--nudge-after N` | read-streak reminder | rejected |
