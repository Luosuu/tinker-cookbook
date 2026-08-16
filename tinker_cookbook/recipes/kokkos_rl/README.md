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

## 2. Auto-annotate and validate candidates

Use a Tinker inference model to propose verification metadata, then execute the
proposal in a fresh Modal sandbox:

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.auto_annotate \
  --candidates data/kokkos/raw_candidates.jsonl \
  --output data/kokkos/validated.jsonl \
  --failed-output data/kokkos/auto_annotation_failures.jsonl \
  --reports-dir data/kokkos/auto_annotation_reports
```

The CLI loads credentials from `.env`. By default it uses OpenAI
`gpt-5.6-terra` through the Responses API with strict Structured Outputs and
medium reasoning effort. Use `--provider tinker` for full
`thinkingmachines/Inkling:peft:262144` with explicit `thinking_effort=0.7`, the
model's recommended renderer, temperature 1.0, and an 8K annotation budget.
Use `--env-file` or the corresponding provider/model/effort flags to override
these settings.

The model proposes:

- `build_targets`: narrow CMake targets affected by the PR;
- `fail_to_pass` / `pass_to_pass`: stable test identifiers;
- `f2p_commands` / `p2p_commands`: exact executable or CTest commands;
- `metadata.f2p_stage = "build"` for expected-compile-failure instances.

Each proposal is schema-checked and run through the same parent, test-only, and
gold-fix transitions used below. Failed command output is fed back to the model
for up to three repair attempts. Only instances that pass real Modal validation
are written to `validated.jsonl`; full attempt records remain in `reports-dir`.
Inference calls are concurrent and rely on the Tinker SDK's own retry behavior.
Successful per-candidate reports are reused on restart; pass `--no-resume` to
force a new proposal and validation run.

`annotate.py` remains available for importing reviewed annotations. Do not put
scoring-test names or gold implementation details into `problem_statement`.

## 3. Validate incrementally (optional local audit)

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

## 5. Train or evaluate on the exported tasks

Both entrypoints read a directory of exported Harbor tasks directly (no
`~/.cache` copy required) and reuse the shared `harbor_rl` loop, so no
Kokkos-specific training code is duplicated.

```bash
# Baseline pass@1 over the exported tasks
uv run python -m tinker_cookbook.recipes.kokkos_rl.eval_kokkos \
  tasks_dir=data/kokkos/phase0/harbor \
  model_name=openai/gpt-oss-120b:peft:131072

# RL training on the same tasks
uv run python -m tinker_cookbook.recipes.kokkos_rl.train_kokkos \
  tasks_dir=data/kokkos/phase0/harbor \
  model_name=openai/gpt-oss-120b:peft:131072 \
  group_size=4 groups_per_batch=8
```

Defaults track the Phase 0 baseline budget (long-context model, 20 turns,
112K-token trajectories). Pin `thinking_effort=0.9` for Inkling rollouts.
