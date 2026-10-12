# Request budget: 40 versus 250

SWE-kokkos-bench v2.4 froze a 40-request budget. On Rivanna, 88% of failed native episodes
ended at that cap. `block-t01` ran native at 40 and at 250 requests, and Pi with the build
check at 250, in the same period (two draws of 100 tasks per arm; the four CUDA tasks are
missing from every arm because the `gpu-mig` node was down).

| comparison | task-paired delta | p | 95% interval |
| --- | ---: | ---: | --- |
| native 250 vs native 40 | +12.1 | 0.001 | +5.3 to +19.0 |
| Pi + build check 250 vs native 40 | +13.7 | 0.001 | +6.3 to +21.6 |
| Pi + build check 250 vs native 250 | +1.6 | 0.55 | −3.7 to +6.8 |

Draws: native 40 at 63.2 and 51.0, native 250 at 65.3 and 75.0, Pi + check 250 at 71.6 and
69.2 (percent solved, 95 tasks with all arms graded).

**The budget is the larger lever.** Raising it adds about twelve points to the same native
loop. The build check's +4.3 at 40 requests (notes/pi_harness/RESULTS.md) mostly disappears
at 250: it was compensating for requests the model did not have.

## What 250-request episodes used

`python -m notes.pi_harness.budget_usage BLOCK`, 373 graded 250-request episodes:

| measure | native 250 (p50 / p99 / max) | Pi + check 250 (p50 / p99 / max) |
| --- | --- | --- |
| requests | 46 / 92 / 94 | 41 / 85 / 102 |
| output tokens | 9,693 / 21,686 / 25,191 | 9,010 / 21,016 / 23,358 |
| cumulative input tokens (max) | 3.26 M | 2.99 M |
| episode seconds | 292 / 1,147 / 1,302 | 323 / 1,589 / 1,787 |
| harness build checks | — | 2 / 8 / 8 |

Every native episode stopped on its own (`model_finished`); Pi episodes stopped on their
own except two that ended on a thinking-only reply. No episode reached the context guard.
Inkling-Small needs roughly 50-100 requests to finish a task here; 40 cut it off midway.

## The long-250 budget

`tinker_cookbook/recipes/kokkos_rl/budget.py` defines it; Rivanna blocks now default to it
(`--budget-profile v2.4` reproduces the frozen benchmark).

| limit | v2.4 | long-250 | basis |
| --- | ---: | ---: | --- |
| model requests | 40 | 250 | 2.5× the observed maximum |
| tool calls | 80 | 500 | one call per request was measured |
| output tokens (episode) | 65,536 | 131,072 | 5× the observed maximum |
| input tokens (episode) | 5,000,000 | 65,536,000 | requests × context window; cannot bind first |
| context per request | — | 262,144 | the model's window; graded, not an error |
| wall seconds | 3,600 | 7,200 | 4× the observed maximum |
| sandbox seconds | 7,200 | 10,800 | wall plus verifier, with margin |
| build checks (Pi) | 8 / 1,200 s | 24 / 3,600 s | the old cap was reached at p99 |

Cost per episode rose with the budget: median cumulative input 0.65 M (native 250) against
0.42 M (native 40), with 96% and 95% of it reported as cached, and median output 9.7 k
against 6.7 k tokens.

## Not changed

- The published v2.4 payloads are immutable; their `task.toml` keeps a 3,600 s agent
  timeout. A dataset release for this budget should export with `agent_timeout_sec=7200` and
  `verifier_timeout_sec=2400` (`export_harbor.export_instance`).
- RFT self-training still fixes 40-request rollouts (`self_train.py`, `rollout_data.eligible`).
  Training at 250 changes rollout length and cost and needs its own decision.
- DSH arms on ConTree cannot use long-250: ConTree sandboxes live at most 3,600 s and the DSH
  proxy enforces its own input cap, so blocks refuse DSH arms with a non-default token budget.
- Leaderboard scores at 40 and 250 requests are not comparable; report the budget name with
  every score.
