# Kokkos dataset construction

This package turns merged Kokkos ecosystem pull requests into SWE-style, test-verifiable
Harbor tasks. The pipeline keeps the gold production patch and held-out test patch out of
the agent image. During grading, protected files are restored before the held-out patch and
task-specific checks are applied.

## 1. Mine candidates

```bash
GITHUB_TOKEN=... uv run python -m \
  tinker_cookbook.recipes.kokkos_rl.dataset.mine \
  --since 2025-12-01 \
  --output data/kokkos/raw_candidates.jsonl \
  --rejections-output data/kokkos/rejections.jsonl
```

The rule-based funnel requires both production and unit-test changes, a bounded diff, and
supported source paths. Repository-specific build profiles live in `ecosystem.py`.

## 2. Annotate and validate

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.dataset.auto_annotate \
  --candidates data/kokkos/raw_candidates.jsonl \
  --output data/kokkos/validated.jsonl \
  --failed-output data/kokkos/auto_annotation_failures.jsonl \
  --reports-dir data/kokkos/auto_annotation_reports
```

The annotation model proposes narrow build targets and exact fail-to-pass and pass-to-pass
commands. Every proposal is then executed in a fresh Modal or Nebius ConTree sandbox. Failed command output
can be fed back for up to three repair attempts. Only candidates satisfying all of these
transitions are written to the validated JSONL:

1. the parent revision configures, builds, and passes selected regression checks;
2. applying only the held-out test patch produces the expected failure;
3. applying the production patch restores all fail-to-pass and pass-to-pass checks.

Per-candidate reports contain the complete annotation and command evidence and make runs
resumable. `annotate.py` is also available for importing reviewed annotations.

To validate CPU and compile-only candidates with ConTree, put
`NEBIUS_SANDBOX_API_KEY` and its authorized `NEBIUS_PROJECT_ID` in `.env`, then add
`--sandbox-backend contree` to the command above. GPU runtime validation remains on Modal
because ConTree SDK 0.3 does not expose GPU selection.

For an optional local audit against an existing checkout:

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.dataset.validate \
  --instances data/kokkos/annotated.jsonl \
  --instance-id kokkos__kokkos-9408 \
  --repo-dir /path/to/kokkos \
  --output data/kokkos/reports/9408.json
```

## 3. Assemble and export

Validated sources can be deduplicated into one SWE-compatible JSONL with `assemble.py`.
Export that JSONL into executable Harbor task directories with:

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor \
  --instances data/kokkos/validated.jsonl \
  --output-dir data/kokkos/harbor
```

Before release, use `release_manifest.py` to combine per-task Oracle and NOP results. A
publishable task must score Oracle reward 1 and NOP reward 0 without infrastructure errors.

## Outputs

- raw JSONL: public PR metadata, issue text, base revision, production patch, test patch;
- rejection JSONL: candidates excluded by static mining rules;
- annotation reports: model proposals and full sandbox command evidence;
- validated JSONL: only instances passing the three construction transitions;
- Harbor export: executable agent environment, protected grader, tests, and Oracle solution.

Do not publish raw queues or failed annotation reports as benchmark examples. They are
provenance and debugging artifacts, not verified tasks.
