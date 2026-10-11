# Kokkos dataset construction

Planned improvements and acceptance criteria are tracked in [TODO.md](TODO.md).

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

## Hosted Tinker annotation and instruction audit

The LLM stages accept `--provider tinker-chat`, using `TINKER_API_KEY` and the hosted
Chat Completions endpoint. GitHub candidate mining and all existing sandbox validation
rules are unchanged.

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.dataset.auto_annotate \
  --provider tinker-chat --model-name thinkingmachines/Inkling-Small:peft:262144 \
  --thinking-effort 0.9 --candidates data/kokkos/candidates.jsonl \
  --output data/kokkos/validated-chat.jsonl \
  --reports-dir notes/experiments/kokkos-annotation-chat --max-candidates 1

uv run python -m tinker_cookbook.recipes.kokkos_rl.dataset.audit_instructions \
  --provider tinker-chat --model-name thinkingmachines/Inkling-Small:peft:262144 \
  --thinking-effort 0.9 --instances data/kokkos/validated-chat.jsonl \
  --reports-dir notes/experiments/kokkos-audit-chat
```

Pass `--checkpoint-url tinker://.../sampler_weights/...` to use saved weights or
`--base-url` to override the endpoint. Hosted inference selects the server renderer;
`--renderer-name` is rejected for this provider. Start with a fresh reports directory:
resume checks bind cached annotations to the provider, model, effort and input digest.
`api_calls/` retains response/usage evidence, including truncated responses which cannot
be accepted as annotations. These are local research artifacts and may contain private
verifier-derived information. Review audit reports before explicitly using `--apply`.
Native `--provider tinker` remains supported.

## Bounded training expansion (Rivanna execution only)

The expansion lane keeps inference on ordinary OpenAI Chat Completions and sends
credential-free execution requests to Rivanna. No Tinker sampler or renderer is
used for these requests. `--provider chat` is also available in `auto_annotate`
and `audit_instructions`: supply `--base-url`, `--model-name`, `--api-key-env`,
and optionally `--provider-parameters '{"reasoning_effort":"high"}'`. Generic
Chat requests send no reasoning or Tinker-specific parameters unless explicitly
provided. Use `TINKER_API_KEY` with the documented Tinker endpoint when desired;
other providers may use a different credential environment variable.

First create a versioned candidate queue, excluding **all** frozen instances
(conservatively including heldout families). Frozen examples are used only for
exclusion and are never included in annotation/audit prompts:

```bash
python -m tinker_cookbook.recipes.kokkos_rl.dataset.expand mine \
  --repo kokkos/kokkos --since 2020-01-01 --until 2026-10-01 \
  --pages 100 --enrich-limit 100 \
  --frozen-instances data/kokkos/SWE-kokkos-bench-v2/instances.jsonl \
  --output-dir /path/to/new/core
```

Repeat with `--repo kokkos/kokkos-kernels`. Rerunning the exact command resumes
pagination/enrichment; `scan_complete` means listing exhausted, **not** that all
pending PRs were enriched or accepted. `report.json` includes both counts.
A source/input digest mismatch requires a new queue directory. Exclusions retain
reasons, unavailable merge commits are rejected, and transport errors stay pending.
The historical kernels `src/` layout is supported. Overlapping patch lines,
linked issues, same merges and explicit backport references are isolated. These
are conservative heuristics, not a proof that all semantic duplicates are found.

Stage the matching `tinker_cookbook` source tree at `<remote-root>/code` on
Rivanna, then run the local coordinator:

```bash
python -m tinker_cookbook.recipes.kokkos_rl.dataset.expand run \
  --candidates /path/to/new/core/candidates.jsonl \
  --frozen-instances data/kokkos/SWE-kokkos-bench-v2/instances.jsonl \
  --output-dir /path/to/new/batch \
  --rivanna-host login.hpc.virginia.edu --rivanna-root /scratch/USER/unique-run \
  --rivanna-base-sif /scratch/USER/compiler-base.sif --rivanna-partition standard \
  --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
  --model thinkingmachines/Inkling-Small:peft:262144 \
  --api-key-env TINKER_API_KEY --env-file .env \
  --provider-parameters '{"reasoning_effort":0.7,"separate_reasoning":true}' \
  --limit 2 --max-attempts 2 --max-tokens 4096
```

`rivanna_bridge.py` currently uses the existing project-specific Rivanna account,
Python environment and Apptainer module paths. The compiler-equipped base SIF
and CPU partition are explicit CLI options. Adapt the account/environment settings
settings for another account before running; there is no fallback to another
execution backend. Each worker requests 4 CPU, 16 GiB, and 30 minutes. Jobs run
serially. The coordinator retains job IDs and polls for completion; pending jobs
are never successes. Infrastructure errors stop annotation repair. It does not
copy API credentials to Rivanna. Execution workers require a Rivanna Slurm
allocation and verify their source digest against the coordinator.

The automated execution profile supports CPU `kokkos/kokkos` with upstream
tests. An experimental CPU kernels lane requires an explicit immutable
`metadata.kokkos_dependency_commit`; construction and export use that same
dependency. Historical executable prefixes are normalized only when declared
by the base CMake wrapper; runtime selectors remain unchanged. Adapter command
timeouts are classified separately and stop automatic model annotation repair.
Compatibility is established by execution for each candidate, never
assumed from a matching version. GPU profiles and generated tests remain isolated.
Statements may be revised at most twice using public issue/PR text only, with
exact source quotations retained. The writer receives neither private patches nor
private audit feedback; each revision receives a fresh independent audit.
Unsupported or uncertain statements remain quarantined. Interrupted revision/audit
calls require reconciliation rather than automatic replay, and cached review
records bind the original instance, public sources and inference configuration. Three-stage execution repeats runtime failures and gold checks three
times; nonzero timeout/resource exits cannot satisfy F2P. The final lane requires
P2P, a clear statement audit plus static leakage checks, a fresh exported image,
two independent NOP/Oracle pairs, and a protected-test tampering probe. Images,
exports, requests and evidence are bound by hashes. `accepted_training.jsonl`
contains only tasks reaching every gate; neither `candidates.jsonl` nor a
construction-only success is eligible for training. Model review does not replace
execution. Receipts cache both acceptance and quarantine; use a fresh batch for
revisiting a quarantined task after fixing a common problem.

Current limits: exact runtime coverage uses the existing command-level collector;
semantic near-duplicate detection and statement leakage review are conservative
heuristics. A successful export needs its own execution evidence; passing offline
unit tests does not certify a dataset. Model usage is retained in `api_calls/`;
monetary costs require the provider's actual billing rates. Nothing is published
or added to the frozen 80/20 experiment by these commands.

The CPU base template is `rivanna_cpu_base.def`. Set its existing agent-server
image path, build it inside a bounded Rivanna allocation with `apptainer build
--fakeroot compiler-base.sif rivanna_cpu_base.def`, and retain the definition and
SIF SHA-256. Its entrypoint matches the existing generic OpenHands image. Do not
use a frozen task image as a generic construction environment.

Before spending annotation calls on historical CPU candidates, the bounded
preflight lane (`RivannaBridge.execute(instance, "preflight")`) configures the
base checkout, reads CMake File API executable/source ownership and CTest
metadata, builds the discovered target, and records its actual gtest inventory.
A missing or empty inventory stops that candidate. Store the returned inventory
as `metadata.runtime_test_inventory`; annotation then rejects P2P filters that
select no observed test. The round-three driver and immutable inputs live in
`artifacts/kokkos-expansion-20261001/round3` in the parent workspace. This explicit
preflight driver is not yet automatically invoked by every legacy CLI entrypoint.

Exported task repositories are initialized empty and fetch only the immutable
base commit with depth one. Checking `git rev-list` alone is insufficient: each
independent trial also verifies that the reference merge object cannot be read.
For compile-stage F2P, a NOP may fail compilation before later checks can run;
acceptance requires an actual compiler diagnostic and every blocked check to
name that build failure. Runtime NOP requires executed F2P failure and passing
P2P. The tampering probe requires the explicit protected-file rejection, and
Oracle requires every planned check to pass. Missing checks, infrastructure
errors, unexplained blocking, and incorrect rewards still quarantine the task.

### Cached source recovery and quota stops

`candidate_recovery.attempted_records(root)` indexes legacy root receipts as well
as nested construction and preflight records; it does not treat candidate queues
as attempts. `recover_source_metadata(candidate, sources)` restores the raw public
source extraction from cached, attributed text without changing the task statement
or reading a reference patch for requirements. Missing source cache stays pending
for supplementation. An out-of-profile candidate is distinct from an attempted
quarantine, and neither category should be inferred merely from a missing field.

For public metadata requests, callers can set
`GitHubAPI(error_records_dir=Path(...))` to retain whitelisted quota response
headers on HTTP errors. Records omit credentials and request headers. They retain
rate-limit category, remaining/reset/retry-after when supplied, and a conservative
retry deadline. The API does not retry automatically. A previous error body with
no recorded reset or retry-after cannot establish that a server waiting window
has elapsed; leave the retry deadline unknown rather than inventing one. Keep
stopped batches immutable and freeze any later candidate cohort separately.

### Header ownership and discovery-only regression

The preflight resolver now uses CMake's actual compile commands and compiler
preprocessor dependencies (`-M`) to associate changed test headers with consuming
translation units and executable targets. It strips object/dependency output flags
before scanning and retains the command-level outcome. Conditional includes follow
the actual target defines and include paths. An ambiguous owner, missing dependency,
or failed scan stays unresolved; it is not reported as an empty runtime suite.

A temporary test overlay makes newly introduced headers discoverable. The overlay
is reversed and baseline CMake configuration restored before normal preflight
builds or lists tests. The `target_discovery` worker action stops after target
resolution and patch restoration: it does not build a target, execute tests, run
annotation, or perform acceptance. Unsupported discovery-only inputs stop before
creating a sandbox. Dependency scanning is bounded to 150 seconds overall and
20 seconds per translation unit. Existing benchmark images are unchanged.

Public-source collection follows explicit issue references in PR titles, bodies,
and human PR comments, then follows linked documents to depth two, with at most
eight root/related documents. Root comments are capped at 100 and truncation is
recorded. Automated coverage comments are excluded from the writer source list.
These bounds improve source coverage; they do not certify semantic completeness
or replace the independent statement audit. Frozen receipts and source caches
remain tied to their original version.

### Versioned requalification after a pipeline repair

`qualification.py` provides an evidence-keyed recovery path without resetting old
attempt counters. A qualification key binds repository/base/patch identity,
public-source manifests, contract coverage, tool versions and execution policy.
`grant_requalification` requires a causal before/after change linked to the old
failure and permits one grant per task in a frozen campaign. Store the new receipt
under its qualification key and retain the parent receipt hash. Do not reuse an
old report directory, delete its pending markers, or interpret an instance's
presence in an old attempted index as a permanent rejection.

The writer receives sanitized public discussions and immutable base-version API
excerpts, never private auditor drafts, hidden tests or reference patches. Public
contract rows contain an observable behavior, source spans and any repository
inference. With `metadata.public_contracts`, revisions must account for every
contract, and audits must supply `contract_findings` with source support and
statement coverage. An unsupported verdict stays `pending_audit_evidence` rather
than becoming a semantic rejection. Raw source caches remain available for
provenance, while patch/solution fences are excluded from the writer pack.

`metadata.execution_policy` fixes command, build, image and grader time limits
before execution. A build tier above 300 seconds requires preflight/timing
provenance; individual command budgets cannot exceed 900 seconds. Target discovery
and nonempty GoogleTest selection remain prerequisites. Runtime filters do not
reduce aggregate compilation work. Timeouts, missing target ownership, unavailable
dependencies and framework failures are recoverable engineering outcomes, not
proof that the bug task is invalid. Signal handling is shared between construction
and exported verification and still requires passing gold/control trials.

Report unique candidate transitions separately from stage attempts. Listing-level
PR coverage is not the denominator for fully qualified task yield. A retry grant
is not an admission: all independent NOP/Oracle repetitions, protected-file checks
and base-history isolation gates still apply.
