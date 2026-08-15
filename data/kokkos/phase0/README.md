# Kokkos Coding-RL Phase 0 Dataset

This directory contains five recent `kokkos/kokkos` instances that passed the
three-stage local validation on 2026-08-15:

1. the parent commit and affected target build successfully;
2. the held-out test patch makes the affected target fail to compile;
3. the production patch makes the target and all configured F2P/P2P checks pass.

`instances.jsonl` is the SWE-compatible source of truth. `harbor/` contains the
same instances exported for `tinker_cookbook.recipes.harbor_rl`. The agent image
is built only from each task's `environment/` directory; `tests/test.patch` and
`solution/gold.patch` remain outside that build context.

All five Harbor images were built and their clean-room graders were checked on
Linux arm64: empty patches score 0, gold production patches score 1, and a test
tampering probe scores 0. These remain feasibility-spike instances rather than a
released benchmark split; model-based leakage and pass@k audits are still needed.

See [`notes/experiments/kokkos_phase0/results.md`](../../../notes/experiments/kokkos_phase0/results.md)
for timings and limitations.
