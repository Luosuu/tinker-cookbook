# Kokkos Coding-RL Dataset Construction

This recipe builds SWE-style, test-verifiable Kokkos tasks for the existing
Harbor RL environment. It deliberately separates data construction from RL:

- JSONL stores the standard `base_commit`, `patch`, `test_patch`,
  `FAIL_TO_PASS`, and `PASS_TO_PASS` fields.
- Harbor export stores hidden grader tests outside the agent image and exposes
  only the repository, problem statement, shell tool, and pre-warmed build tree.
- The grader rejects changes to tests, CMake registration, or CI before restoring
  pristine tests and injecting the held-out test patch.

## 1. Mine raw candidates

Use a GitHub token to avoid the anonymous REST rate limit:

```bash
GITHUB_TOKEN=... uv run python -m tinker_cookbook.recipes.kokkos_rl.mine \
  --since 2025-12-01 \
  --output data/kokkos/raw_candidates.jsonl \
  --rejections-output data/kokkos/rejections.jsonl
```

The rule-based funnel requires production and unit-test changes, a 5--500 line
diff, and at least one non-GPU production path.

## 2. Annotate the Phase 0 candidates

For each selected row, fill in:

- `build_targets`: narrow CMake targets affected by the PR;
- `fail_to_pass` / `pass_to_pass`: stable test identifiers;
- `f2p_commands` / `p2p_commands`: exact executable or CTest commands;
- `metadata.f2p_stage = "build"` for expected-compile-failure instances.

Do not put scoring-test names or gold implementation details into
`problem_statement`.

## 3. Validate incrementally

Clone Kokkos once, then reuse it as the git worktree source:

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.validate \
  --instances data/kokkos/annotated.jsonl \
  --instance-id kokkos__kokkos-9408 \
  --repo-dir /path/to/kokkos \
  --output data/kokkos/reports/9408.json
```

The validator builds the parent, checks P2P, applies only tests and checks F2P
failure, applies production changes, then requires all F2P/P2P commands to pass.
F2P failures are repeated three times by default to screen flaky tests.

## 4. Export validated Harbor tasks

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.export_harbor \
  --instances data/kokkos/validated.jsonl \
  --output-dir ~/.cache/harbor/tasks/kokkos-rl
```

Load the result with `load_harbor_tasks("kokkos-rl")`, then use the existing
`harbor_rl` training or evaluation scripts.
