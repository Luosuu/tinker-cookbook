# Dataset TODOs

## Collect complete F2P and P2P results in Harbor verifiers

Status: planned; implementation and dataset regeneration are deferred.

The exported verifier currently builds the selected targets, then executes
fail-to-pass (F2P) commands followed by pass-to-pass (P2P) commands, stopping at
the first failure. This gives a binary reward but hides later test outcomes.
Serial execution does not require stopping on failure: the immediate goal is
complete diagnostic coverage, not additional concurrency.

- [ ] Update `export_harbor.py` and the runtime result collector to continue
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
- [ ] Keep binary reward strict: all required checks must pass. A failed build,
  blocked required check, zero selected tests, or invalid result must never
  produce reward 1. Preserve clean-room and hidden-test protections.
- [ ] Persist detailed results through Harbor evaluation logs and experiment
  archives, alongside reward, task identity, verifier version, and source hashes.
- [ ] Add execution tests covering F2P failure followed by P2P execution,
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
