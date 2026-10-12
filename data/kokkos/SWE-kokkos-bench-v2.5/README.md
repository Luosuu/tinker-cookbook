# SWE-kokkos-bench v2.5

The same 100 Kokkos ecosystem regression-repair tasks as v2.4, re-released for a 250-request
evaluation budget. Agents modify production code in `/workspace/repo`; hidden tests and
reference solutions are unavailable during solving. A successful repair must pass every
required regression and preservation check.

All 100 tasks passed fresh NOP/Oracle acceptance on Rivanna Slurm/Apptainer for this
release; `validation-manifest.json` records task digests, runtime image digests, isolation
checks and control results.

## Changes from v2.4

Only task metadata changed. Instructions, hidden tests, verifiers, environments and reference
patches are byte-identical to v2.4.

- `[task] version` is `2.5.0`.
- `[agent] timeout_sec` is 7200 (was 3600) and `[verifier] timeout_sec` is 2400 (was 1200).
- `[metadata]` names the evaluation budget: `budget = "long-250"`, `max_model_requests = 250`,
  `max_tool_calls = 500`, `max_context_tokens = 262144`.
- `budget.json` gives the complete budget.

## Evaluation budget

| limit | v2.4 | v2.5 (`long-250`) |
| --- | ---: | ---: |
| model requests | 40 | 250 |
| tool calls | 80 | 500 |
| output tokens per response | 16,384 | 16,384 |
| output tokens per episode | 65,536 | 131,072 |
| tool result characters | 16,384 | 16,384 |
| context per request | — | 262,144 (stop and grade) |
| agent wall time | 3,600 s | 7,200 s |

Why: with Inkling-Small on Rivanna, 88% of failed episodes at 40 requests ended at the cap,
and in a same-period comparison the same native agent solved 12.1 points more tasks with
250 requests (p = 0.001). At 250 requests every episode stopped on its own by request 102,
so the cap itself no longer binds; the other limits sit well above measured use.

When an episode reaches the context limit, stop issuing requests and grade the repository as
it is; do not count it as an infrastructure error.

## Execution profiles

96 tasks use CPU environments. Tasks 8989, 9147, 9159 and 9328 require CUDA. The portable
CUDA recipes target CUDA 12.8.1 and Ada SM89; Rivanna validation uses a separately recorded
CUDA 12.9.1 / Blackwell SM120 translation with an explicit GPU allocation (for this release,
a full RTX PRO 6000; for v2.4, a MIG slice). This is an execution profile difference, not
evidence of testing every GPU architecture.

## Evaluation reporting

Report scores against the immutable v2.5 version and its task digests, with the budget name.
**Scores at v2.4 (40 requests) and v2.5 (250 requests) are not comparable.** Report model
identifier, agent, sampling settings, task subset, trial count, valid scores, errors, and
token/tool cost separately. Small subsets are diagnostic and are not the full benchmark score.

## Sources and attribution

Tasks derive from public pull requests in [Kokkos Core](https://github.com/kokkos/kokkos),
[Kokkos Kernels](https://github.com/kokkos/kokkos-kernels), and
[PyKokkos](https://github.com/kokkos/pykokkos). Each task's metadata records its repository
and source revisions. Upstream source, patches and tests retain their respective licenses and
attribution. The dataset construction code in the
[cookbook fork](https://github.com/Luosuu/tinker-cookbook/tree/science-rl) is Apache-2.0; this
mixed-source dataset retains the `other` license classification of previous releases.

Earlier releases remain available under their Harbor tags (`v2.4`, `v2.3`). Cite the Harbor
tag and immutable content digest rather than inferring the version from a metadata field.
