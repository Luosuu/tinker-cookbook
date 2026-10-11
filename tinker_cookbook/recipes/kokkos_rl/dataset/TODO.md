# Dataset TODOs

## Collect complete F2P and P2P results in Harbor verifiers

Status: command-level complete-result prototype implemented; candidate validation in progress. Existing snapshots remain unchanged. Individual case extraction, dependency annotations, deduplication, and bounded test concurrency remain follow-up work.

The previous exported verifier builds the selected targets, then executes
fail-to-pass (F2P) commands followed by pass-to-pass (P2P) commands, stopping at
the first failure. This gives a binary reward but hides later test outcomes.
Serial execution does not require stopping on failure: the immediate goal is
complete diagnostic coverage, not additional concurrency.

- [x] Update `export_harbor.py` and the runtime result collector to continue
  independent, runnable checks after ordinary test failures, including P2P
  checks when F2P fails. Preserve prerequisite ordering and protected-file checks.
- [ ] Record structured build and test outcomes: passed, failed, skipped,
  blocked by prerequisites, and infrastructure/error states. Record the cause
  of blocked checks; never infer failure or success for an unexecuted test.
- [ ] Where supported, collect individual GoogleTest/CTest/pytest case results
  and classify them against F2P/P2P membership after execution. Keep compile-only
  checks and shell checks explicit; do not label command counts as unit-test counts.
- [ ] Avoid redundant builds and duplicate test execution when safe. Audit
  overlapping F2P/P2P commands and missing P2P coverage; changing test membership
  requires separate review and validation, not an automatic inference.
- [x] Keep binary reward strict: all required checks must pass. A failed build,
  blocked required check, zero selected tests, or invalid result must never
  produce reward 1. Preserve clean-room and hidden-test protections.
- [ ] Persist detailed results through Harbor evaluation logs and experiment
  archives, alongside reward, task identity, verifier version, and source hashes.
- [x] Add execution tests covering F2P failure followed by P2P execution,
  multiple F2P checks, P2P regression, build failure, zero-test selection,
  malformed output, and protected-file tampering.
- [ ] Export a new candidate dataset snapshot; retain existing release and
  experiment snapshots unchanged. Revalidate Oracle=1 and NOP=0 without
  infrastructure errors for every updated task before publication or training.
- [ ] Compare old/new grading on saved candidates without new model sampling.
  Explain score differences and measure build/test time and resource use.
  Treat bounded concurrency of independent runtime tests as a subsequent
  optimization; do not concurrently mutate a shared build directory.

Acceptance: an F2P assertion failure no longer prevents independent P2P checks
from being reported; prerequisite failures remain distinguishable from executed
test failures; detailed results reconcile with the strict overall reward.

References for the run-then-classify approach:

- [SWE-bench Python evaluation script generation (v4.1.0)](https://github.com/SWE-bench/SWE-bench/blob/v4.1.0/swebench/harness/test_spec/python.py)
- [SWE-bench grading](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py)
- [pytest failure handling](https://docs.pytest.org/en/stable/how-to/failures.html)


### Command-level prototype

`export_harbor.py` embeds the standard-library `verifier_results.py` collector.
It runs protected-file preparation and the build before F2P/P2P commands. Ordinary
assertion failures no longer stop later checks. Failed prerequisites block later
checks with their cause recorded. The collector writes `results.json` atomically
after each check, retains command stdout/stderr, records elapsed time and verifier/
test-patch hashes, and awards one only when every required command passes.
Harbor persists the structured report with its grading log and rejects incomplete
or infrastructure/coverage-error reports. Reports explicitly count commands.

Execution tests cover multiple F2P checks, continuing P2P, P2P regression, failed
builds, malformed/zero-test output, and protected-file tampering. A paired CPU
sandbox experiment compares old/new NOP, Oracle, and saved-candidate results on
unchanged images. This is validation of a candidate workflow, not publication of
a new complete dataset release. Full-task Oracle/NOP revalidation remains required.

Parallelism is intentionally not enabled by this prototype. Commands may mutate
a shared build tree or depend on preceding commands. A subsequent optimization
must declare safe runtime groups and keep setup, builds, and dependency boundaries
serial; measure runtime-only time before choosing a concurrency level.

## Automatic training expansion (2026-10-01 development batch)

- [x] Add bounded, resumable historical mining with explicit pending counts and
  exclusion reasons; paginate REST file lists and support historical kernels `src/`.
- [x] Add conservative frozen-family and patch overlap exclusion before prompting.
- [x] Add provider-neutral Chat Completions options to annotation and auditing.
- [x] Add a local inference / credential-free Rivanna Slurm execution bridge.
- [x] Distinguish infrastructure exits from expected F2P failure and repeat gold checks.
- [x] Implement fail-closed construction/audit/export/independent acceptance gates.
- [x] Complete a successful new-task end-to-end acceptance on Rivanna, including
  fresh exported NOP/Oracle repeats and protected-test tampering. Implementation
  and unit tests alone do not close this item.
- [ ] Generalize Rivanna account/image settings and add historical kernels dependency
  version profiles; validate kernels and accelerator tasks before enabling them.
- [ ] Strengthen semantic family detection, stage-resume behavior after interrupted
  model calls, and instruction leakage checks beyond the conservative initial lane.
- [ ] Reach 100 **additional fully accepted** training tasks. This is a milestone,
  not an extrapolation from the candidate funnel. Keep the frozen 80/20 data intact.

The current UVA endpoint reports Slurm cluster name `shen`; the Rivanna worker
recognizes that verified identifier as well as `rivanna`. All execution evidence
must come from the configured UVA allocation. Local tests are development checks.

### Bounded continuation

- Public-source revision and independent re-audit are implemented with a maximum
  of two revisions. Pending request markers prevent blind replay after interruption.
- Historical CPU kernels can now specify an immutable Kokkos dependency, shared
  by construction and export. This is an experimental lane until real acceptance.
- The follow-up batch fixes four candidate IDs before execution; a statement-only
  revision never invalidates or silently replaces the earlier construction record.
  Missing final export evidence still prevents training admission.

The bounded continuation completed with no new admitted tasks: public-source
revision still failed the contract audit for one revisited task; one additional
candidate passed construction but lacked a supported statement; another selected
zero runtime tests; two historical kernels builds hit the command time limit.
These outcomes do not establish an end-to-end successful release. Adapter timeout
classification and the no-extra-model-retry rule now have regression coverage.
Historical kernels target normalization follows the base CMake prefix declaration
and preserves runtime selectors. This later repair is regression-tested, not a
claim of full candidate revalidation. All new checks ran in Rivanna allocations.

### Source and inventory gated continuation

The fixed two-candidate round admitted `kokkos__kokkos-2842` after construction,
public-source statement revision and independent audit, then fresh exported
NOP/Oracle/NOP/Oracle/tampered-NOP trials with rewards 0/1/0/1/0 (Rivanna job
20716827). Both Oracle runs passed every check; NOP exhibited the declared
compile-stage regression; the tampered run explicitly rejected protected edits.
`kokkos__kokkos-2841` remains quarantined for a baseline compiler compatibility
failure before any annotation call. The original full-clone export failure is
preserved; depth-one acquisition repaired its reference-object leakage.

Round-three source-only regression jobs culminated in 73 passing checks on
Rivanna. Source extraction retains reproducer variants and raw fragment offsets;
preflight records actual CMake target ownership and gtest names before inference.
The admitted JSONL and complete evidence remain in the parent workspace under
`artifacts/kokkos-expansion-20261001/round3`; the frozen 100-task data is unchanged.
This establishes one completed admission, not the 100-new-task milestone.
