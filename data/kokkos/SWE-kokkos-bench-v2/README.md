# SWE-kokkos-bench v2.0

SWE-kokkos-bench is a collection of 100 verifier-backed coding tasks mined
from merged pull requests in the Kokkos ecosystem. Version 2.0 expands the
published 30-task v1.0 snapshot with 70 additional `kokkos/kokkos` tasks.

```bash
uvx harbor run -d luosuu/SWE-kokkos-bench@v2.0 -a <agent> -m <model>
```

## Contents

- 90 tasks from `kokkos/kokkos`
- 8 tasks from `kokkos/kokkos-kernels`
- 2 tasks from `kokkos/pykokkos`
- 76 compile-failure tasks and 24 runtime-test-failure tasks

Each task starts from the parent of a merged pull request. The agent sees the
issue statement and repository checkout, while the verifier restores protected
test files, injects the held-out test patch, and runs the task-specific build
and regression commands without network access.

## Validation

An accepted task must demonstrate the following transitions in a fresh Modal
sandbox:

1. the parent checkout configures and builds;
2. the held-out test patch fails on the parent;
3. the production patch restores the build and held-out checks;
4. the exported Harbor Oracle receives reward 1;
5. the exported Harbor NOP agent receives reward 0.

The 30 unchanged v1.0 tasks retain identical Harbor digests and reuse their
published Oracle/NOP evidence. The 70 additions are validated separately in
the v2.0 release evidence. See `validation-manifest.json` and the experiment
report under `notes/experiments/SWE-kokkos-bench/` for per-task provenance and
aggregate results.

## Dataset fields

`instances.jsonl` contains standard SWE-bench fields (`repo`, `instance_id`,
`base_commit`, `problem_statement`, `patch`, `test_patch`, `FAIL_TO_PASS`, and
`PASS_TO_PASS`) plus executable configuration, build targets, commands, and
toolchain metadata. `manifest.json` records construction provenance;
`dataset.toml` is the Harbor registry manifest.

## Limitations

The tasks are derived from public pull requests, so models may have encountered
the source changes during pretraining. Most additions use compile failures as
their verifier signal, and the collection is weighted toward Kokkos Core rather
than the entire ecosystem. Scores should therefore be reported with the exact
dataset version, agent scaffold, model identifier, and rollout budget.

Kokkos and its ecosystem repositories retain their upstream licenses. This
dataset packages derived issue statements, patches, and build metadata for
research and evaluation; consult each task's source repository for the
applicable license and attribution.
