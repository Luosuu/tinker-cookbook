# Inkling-Small on SWE-kokkos-bench: where the score goes

Analysis of the frozen baseline corpus `notes/genesis_archive/round01/trajectories.json`
against `thinkingmachines/Inkling-Small:peft:262144` (400 episodes = 100 tasks x 4
samples, score **50.0%**). Produced by
[`analyze_trajectories.py`](analyze_trajectories.py); reproduce with:

```bash
python notes/dsh_inkling/analyze_trajectories.py \
  notes/genesis_archive/round01/trajectories.json --tasks
```

This is the evidence base for the DSH harness iteration loop. It is an analysis of an
existing corpus, not a new benchmark result: the corpus was collected with the
cookbook's bash-only harness, so its absolute score is a property of *that* harness.

## Headline

The score is **not** limited by the model failing to understand the tasks, and it is
**not** limited by looping. It is limited by **episodes that run out of turns**. The
single largest bucket is 166/400 (41.5%) of episodes that hit the 40-turn cap and still
would not have passed.

| stop reason | n | share | pass rate | mean turns | mean sampled tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `completed` | 130 | 32.5% | **85.4%** | 29.2 | 6,019 |
| `max_turns` | 255 | 63.7% | **34.9%** | 40.0 | 8,174 |
| `parse_error` | 2 | 0.5% | 0.0% | 28.0 | 7,058 |
| no reason (infra error) | 13 | 3.2% | 0.0% | 23.4 | 0 |

When the model finishes on its own it almost always succeeds. The 50-point gap between
these two rows is the harness's addressable space.

## Where the score comes from

```
episodes=400  passed=200  score=50.0%
  PASS   completed      n=111 (27.8%)
  PASS   max_turns      n= 89 (22.2%)
  fail   max_turns      n=166 (41.5%)   <- largest bucket
  fail   completed      n= 19 (4.8%)
  fail   no reason      n= 13 (3.2%)
  fail   parse_error    n=  2 (0.5%)
```

Splitting the capped episodes by whether they produced any patch at all:

| capped group | n | pass rate |
| --- | ---: | ---: |
| produced a patch | 175 | 50.9% |
| produced **no** patch | 80 | **0.0%** |

Two mechanically different problems:

1. **80/400 (20%) episodes end with an empty patch.** These are a hard 20-point loss
   with zero upside. They are the cheapest target in the whole corpus.
2. **175/400 (44%) episodes do produce a patch but run out of turns**, and only half of
   those pass. Here the budget is nearly sufficient, so the lever is efficiency, not
   capability.

## Why episodes exhaust the budget

### Not looping

Repeated-command share across the 80 empty-patch capped episodes is **0.025** mean, with
**0/80** episodes above 50% repeats. The model is not stuck in a loop; it is doing
distinct work and simply never converging. A repetition detector would not help.

### Overwhelmingly reading

| command kind | calls | share |
| --- | ---: | ---: |
| read/search | 10,725 | **75.8%** |
| other | 1,973 | 13.9% |
| build | 1,425 | 10.1% |
| tooling | 33 | 0.2% |

### Writing arrives late, or never

| group | ever wrote | mean turn of first write |
| --- | ---: | ---: |
| `completed` | 97.7% | **9.5** |
| `max_turns` | 82.0% | **21.8** |

Among capped *with-patch* episodes, **39/173 first touched the repo at turn ≥ 30** — i.e.
they spent three quarters of the budget before editing, leaving almost no turns to
iterate on the fix or run the build.

In the empty-patch group, **44/80 never wrote anything anywhere**, and of the writes that
did happen, **64 targeted `/tmp`** (scratch test programs) versus **1 targeting the repo**.
That is a concrete scaffold defect: the model explores, probes in scratch space, and
never commits an edit to `core/` before the cap.

### What the final turns look like

The last command in empty-patch episodes is almost always a read (`grep`, `sed -n`,
`find`, `ls`). The last recorded thought is typically mid-deliberation about *where* to
place the change (`"We'll insert to_array overloads inside namespace Kokkos after the get
functions, before closing namespace..."`). The model has a plan and no budget left to
execute it.

## Task-level context

- 100 tasks; **31 always pass**, **32 always fail**, **37 are mixed**.
- The 37 mixed tasks are where harness iteration can show a signal; the 32 always-fail
  tasks may be capability or task-difficulty limits and should be reported separately
  rather than averaged in.

## Audit flags

| flags | n | share |
| --- | ---: | ---: |
| none | 300 | 75.0% |
| `empty_patch` | 73 | 18.2% |
| `answer_lookup` | 15 | 3.8% |
| `answer_lookup,empty_patch` | 6 | 1.5% |
| `verifier_access` | 2 | 0.5% |
| other | 4 | 1.0% |

Answer-lookup is ~5% of episodes. It is a correctness hazard for *training* but not the
main score lever, and any harness change must not enlarge it.

## Harness changes this argues for

Ordered by expected value per unit of work. Each is a hypothesis for the DSH harness,
not a result — DSH is the thing being iterated, so these are its first candidate knobs.

1. **A write-forcing / commit-the-edit mechanism.** The dominant failure is areading
   loop that never edits. A harness-side nudge (for example a turn-budget checkpoint that
   requires a repository edit before continuing, or a staging instruction that has the
   agent apply its best hypothesis early and iterate on a build) targets the 20%
   empty-patch loss and the late-first-write population simultaneously.
2. **Keep scratch probes out of the critical path.** 64 writes went to `/tmp`. If
   scratch test programs are valuable they should be cheaper (e.g. a single reusable
   compile harness), and if they are not, the scaffold should not invite them.
3. **Make the build loop cheap.** 10.1% of calls are builds, but the model measures
   build cost itself and defers editing because of it. A pre-warmed build or a documented
   fast incremental command would reduce the perceived cost of the edit-build-iterate
   cycle.
4. **Turn budget and efficiency are separate experiments.** Because capped-with-patch is
   50.9%, either raising the cap or compressing the exploration phase could move it. They
   are different harness edits and should be measured independently against the same
   frozen task split.
5. **Do not bother with a repetition breaker.** The data says looping is not the problem.

## Method notes and limits

- The write detector treats `/dev/null` and fd-duplication redirects as reads
  (`grep ... 2>/dev/null` is a read). Without that, 69 empty-patch episodes would be
  misclassified as having written. Pinned by
  [`analyze_trajectories_test.py`](analyze_trajectories_test.py).
- The corpus is self-training collection data from one harness with a fixed 40-turn cap.
  It cannot say what a *different* turn budget would yield.
- `logical_score` is the recorded reward, used here as ground truth. 13 episodes carry no
  reason and no score; they are infrastructure failures, not model failures, and are
  reported separately rather than counted as losses.
- 2 parse errors are too few to support any conclusion about parse handling.
