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

## Released as SWE-kokkos-bench v2.5

[v2.5](https://hub.harborframework.com/datasets/luosuu/SWE-kokkos-bench) (revision 8, version
`2cda7fb0-6e1e-4699-9af0-8a157f403b61`, content `sha256:c253fbcb19…`, tags `v2.5`, `2.5.0`,
`latest`) is v2.4 with metadata-only changes: task version 2.5.0, agent timeout 7,200 s,
verifier timeout 2,400 s, and `[metadata] budget = "long-250"`; `budget.json` ships with it.
All 200 NOP/Oracle controls passed again on Rivanna before publication, and the published
payload was read back byte-identical. The payload is in `data/kokkos/SWE-kokkos-bench-v2.5`.
v2.4 is unchanged under its own tags.

RFT self-training (`self_train`, default `budget=long-250`, `sandbox_backend=apptainer` on
Rivanna) and the evaluators (`eval_kokkos`, `eval_kokkos_openai`, `eval_kokkos_tinker_chat`,
`eval_kokkos_pi`, Rivanna blocks) now default to this budget and dataset. Guarded payloads are
graded as published rather than regenerated.

## Still at 40 requests

- The Nebius sweep, `baseline_recovery` and the donor scripts reproduce 40-request runs.
- DSH arms on ConTree: ConTree sandboxes live at most 3,600 s and the DSH proxy enforces its
  own input cap, so blocks refuse DSH arms with a non-default token budget. Other ConTree
  users get a 3,600 s sandbox under long-250; rare episodes longer than that plus grading end
  as infrastructure errors.
- Leaderboard scores at 40 and 250 requests are not comparable; report the budget name with
  every score. The v2.4 leaderboard stays bound to v2.4.
