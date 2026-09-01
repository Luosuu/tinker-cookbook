# Kokkos Coding-RL Dataset: Phase 0 Plan

## SWE-kokkos-bench v2.3 paired 20-task model evaluation (2026-09-01)

Research question: on the same 20 Kokkos tasks used for the prior Terra comparison, do the
verifier-aligned v2.3 instructions change pass@1 for GLM-5.3 and Inkling-Small relative to their
pre-v2.3 trajectories?

Hypothesis: clarifying observable API and compatibility requirements should help both models on
previously underspecified tasks, with the largest gains on failures caused by guessing the wrong
surface contract. A 20-task pass@1 run is noisy, so task-level flips and the historical
Inkling-Small eight-sample success frequencies are more informative than the aggregate delta alone.

Experiment design:

1. Use the exact 20 task IDs from the GPT-5.6-Terra comparison and the local v2.3 task payloads.
2. Run one rollout per task for `zai-org/GLM-5.3:peft:262144` with the automatically recommended
   `glm5_3_max_reasoning` renderer.
3. Run one rollout per task for `thinkingmachines/Inkling-Small:peft:262144` with the automatically
   recommended `tml_v0` renderer and explicit thinking effort 0.9.
4. Hold temperature (1.0), turn/tool/token budgets (40 turns, 80 tool calls, 64K sampled tokens),
   network isolation, and Modal concurrency (4 per model) fixed. Use a 900-second grader timeout
   so slow Kokkos compilation is not scored as a model failure.
5. Retry infrastructure errors only, then compare paired pass/fail outcomes with historical
   pre-v2.3 records on the same task IDs.

Controls and success criteria: require 20 valid graded results per model and report infrastructure
errors separately. Do not retry incorrect solutions. Record exact model IDs, renderer, effort,
commit, task list, and result paths. Treat any aggregate movement as directional unless supported
by consistent task-level flips, because one sample per task has substantial sampling variance.

Outcome: both runs produced 20 valid graded trajectories with zero final errors. GLM-5.3 improved
from 3/20 to 12/20 (nine positive and zero negative paired flips), while Inkling-Small improved
from 7/20 to 13/20 (seven positive and one negative paired flips). See
`notes/experiments/SWE-kokkos-bench/v2.3-paired20/report.md` for task-level results and caveats.

## SWE-kokkos-bench v2.3 release validation and Terra cost audit (2026-08-31)

Research questions: do all 100 verifier-aligned v2.3 task payloads still satisfy the Harbor
Oracle=1/NOP=0 release gate under their new digests, and why did the 20-task GPT-5.6-Terra run
cost about $7 despite using the balanced model tier?

Hypotheses: because v2.3 changes only public instructions and metadata, every Oracle and NOP
transition should remain unchanged. Terra's observed cost is expected to come primarily from
hundreds of stateful tool-use turns repeatedly billing a growing context, amplified by unbounded
shell output and high reasoning effort; the evaluator currently obscures this by recording only
the final turn's input usage instead of cumulative input and cache details.

Experiment design:

1. Run all 100 local v2.3 tasks through Harbor's Oracle and NOP agents on Modal, concurrency 6,
   retaining one result per new task digest.
2. Require 100 completed Oracle rewards of 1 and 100 completed NOP rewards of 0 with no task or
   infrastructure errors; retry infrastructure failures only.
3. Generate a new digest-bound validation manifest from those exact results and publish the local
   dataset publicly with immutable tag `v2.3` only after the gate passes.
4. Reconstruct cumulative request input, output, reasoning, and tool-output volume from the prior
   20 unique Terra transcripts. Compare the implied charge with current official token prices and
   the user's observed bill.
5. Fix usage accounting and add bounded tool output, cumulative input/cost budgets, and lower-cost
   defaults while preserving explicit CLI overrides for full-budget evaluations.

Controls and success criteria: do not reuse v2.2 validation evidence for v2.3 digests; do not
publish on partial success; do not retry incorrect model answers as infrastructure errors. A cost
optimization is accepted only if its resolved config and token accounting are saved, tests cover
the accounting/truncation behavior, and any changed evaluation budget is reported with scores.

## Inkling instruction-quality audit (2026-08-31)

Research question: do the 100 released SWE-kokkos-bench v2 task statements expose enough of the
observable contract for a capable coding model to satisfy their held-out verifiers without relying
on upstream history or guessing private symbol names?

Hypothesis: most statements are adequate, but a model-assisted comparison against the private test
and reference patches will identify a small set of API-name mistakes, omitted compatibility modes,
underspecified interfaces, and CI-only requests that materially depress measured model performance.

Experiment design: run `thinkingmachines/Inkling:peft:262144` over all 100 tasks with explicit
thinking effort 0.99, temperature 1.0, and one structured audit per task. Give the auditor the
current statement plus private test/reference patches as evidence, but require a concise behavioral
contract that does not reveal test identities or reference implementation details. Review all
proposed revisions and apply only statements that improve agreement with the verifier. Regenerate
the executable task payloads from the updated source JSONL.

Controls and success criteria: preserve statements assessed as clear; do not change gold patches,
test patches, verifier commands, or repository revisions; reject outputs that disclose hidden-test
details or prescribe an unnecessary implementation; keep invalid CI/test-only tasks explicitly
separate rather than disguising them as production fixes. The audit is complete when all 100 tasks
have parseable reports, every applied statement matches its verifier's observable contract, and
source JSONL, metadata, and exported instructions agree.

## Research question

Can recent host-backend Kokkos bug-fix pull requests be converted into stable,
containerized, test-verifiable coding tasks whose incremental verification cycle
is cheap enough for agentic RL?

## Hypothesis

Kokkos is a good fit for SWE-style RL data because its dependencies are small and
many pull requests add focused GoogleTests. A shared C++20 image, preconfigured
build tree, ccache, target-narrowed rebuilds, and clean-room test injection should
make recent Serial/OpenMP tasks reliable enough for GRPO.

## Phase 0 experiment

1. Mine merged pull requests targeting `kokkos/kokkos:develop`.
2. Keep changes that touch both production code and unit tests, are 5--500 changed
   lines, and are not confined to CUDA/HIP/SYCL code.
3. Select 3--5 recent C++20 bug fixes and annotate their build targets and exact
   F2P/P2P commands.
4. For each candidate, run one incremental worktree through:
   - parent commit build and P2P baseline;
   - test-only patch, which must make every F2P command fail;
   - production patch, which must make all F2P and P2P commands pass;
   - repeated F2P execution to screen for flakiness.
5. Export validated instances in both SWE-compatible JSONL and Harbor task format.
6. Measure configure, initial build, test-only rebuild, gold rebuild, and test times.

## Controls and baselines

- Reject test-only, docs-only, CI-only, large-refactor, and GPU-only changes.
- Run the affected target's baseline tests before injecting the new tests.
- Preserve the PR's original patch as the gold reference, but hide the test patch
  from the agent during rollouts.
- At scoring time, reject test/CMake/CI edits, restore pristine tests, inject the
  held-out test patch, and run both F2P and P2P commands.

## Success criteria

- At least 3 of the first 5 candidates complete the three-step transition.
- No F2P command is flaky across three validation repetitions.
- Median incremental verification time is at most five minutes.
- Exported tasks load through the existing Harbor RL recipe without schema changes.
- The prompt contains no gold implementation details or scoring-test identities.

## Scale-up decision

Proceed to bulk mining only if the Phase 0 success criteria hold. Otherwise first
address the dominant failure mode: target discovery, compiler-era mismatch,
compile latency, or weak/flaky tests.

## Automated annotation experiment

Research question: can a Tinker inference model replace manual discovery of
Kokkos build targets and F2P/P2P commands without weakening the executable data
quality gate?

Hypothesis: the PR's split test/code patches contain enough CMake and GoogleTest
context for a coding model to propose a narrow verification plan. Model output
alone is not trusted: a candidate is accepted only after a fresh Modal sandbox
reproduces parent success, test-only failure, and gold-patch success.

Experiment design:

1. Ask the model for a strict, allowlisted JSON annotation.
2. Run the proposal in Modal and retain capped command logs.
3. On failure, return the last failing commands and logs to the model for at most
   three total proposals.
4. Compare automatic annotations with the five Phase 0 human annotations, then
   run a held-out batch of newly mined PRs.

Controls: annotations cannot overwrite commits, patches, problem statements, or
repository identity; test-stage plans require an executable F2P command; and all
accepted rows pass the same three-stage validator. Report annotation yield,
attempt count, failure category, wall time, and agreement with Phase 0 targets.

Success criteria: reproduce at least four of five Phase 0 instances without
manual edits, introduce no false accepts, and achieve at least 50% validated
yield on the first held-out candidate batch.

## Phase 0 model baseline

Research question: can unmodified GPT-OSS 120B and Inkling-Small complete these
five Kokkos tasks through the same bash-tool and hidden-grader loop intended for
RL?

Experiment design:

1. Use the long-context Tinker variants: `openai/gpt-oss-120b:peft:131072` and
   `thinkingmachines/Inkling-Small:peft:262144`.
2. Use temperature 1.0, at most 40 agent turns, at most 80 tool calls, at most
   65,536 sampled tokens over the trajectory, and at most 16,384 sampled tokens
   per turn. Pin Inkling thinking effort to 0.9. The limits were raised from the
   initial 20-turn smoke after Inkling reached the grader while still completing
   a coherent multi-file fix; the sampled-token cap continues to bound cost.
3. Run one rollout per model-task pair through a fresh Modal sandbox.
4. Record binary hidden-test reward, task errors, elapsed time, turns, and full
   trajectory text.

This first pass is a pipeline and pass@1 baseline, not a statistically stable
model comparison. If rewards are mixed, follow with multiple samples per task to
measure pass@k and estimate the within-group reward variance available to GRPO.

Success criteria: all ten rollouts reach the grader without infrastructure
errors, and the result artifacts are sufficient to diagnose failures even if
both models receive zero reward.

## SWE-kokkos-bench v2 new-70 model baselines

Research question: how do Full Inkling, Inkling Small, and GPT-OSS 120B perform
at pass@1 on the 70 Kokkos Core tasks added in SWE-kokkos-bench v2.0?

Experiment design:

1. Define the evaluation set as the exact set difference between the published
   100-task v2.0 snapshot and the unchanged 30-task v1.0 snapshot.
2. Reuse each model's historical baseline configuration so its new-70 score can
   be combined with the old-30 score without a configuration mismatch:
   - Full Inkling: `thinkingmachines/Inkling:peft:262144`, effort 0.99,
     temperature 1, 40 turns, 60 tool calls.
   - Inkling Small: `thinkingmachines/Inkling-Small:peft:262144`, effort 0.9,
     temperature 1, 40 turns, 80 tool calls.
   - GPT-OSS: `openai/gpt-oss-120b:peft:131072`, temperature 1, 20 turns,
     40 tool calls.
3. Keep the shared limits at 16,384 sampled tokens per turn, 65,536 sampled
   tokens per trajectory, and 114,688 total trajectory tokens.
4. Run one representative smoke task per model before launching the full set.
5. Run one rollout per model-task pair in fresh Modal sandboxes, retaining full
   trajectories, rewards, elapsed time, and errors.

Controls and success criteria: use the same v2.0 task payloads that passed the
Oracle/NOP release gates; do not retry wrong answers; investigate and rerun only
infrastructure failures. A reportable result requires all 210 rollouts to reach
the grader or to have any residual infrastructure failures explicitly separated
from model failures. Report new-70 scores and, where settings match, combined
100-task scores with exact model identifiers and effort values.
# Nebius ConTree sandbox integration smoke test (2026-08-17)

- Research question: can ConTree replace Modal/SandboxFusion at the sandbox boundary used by
  Code RL and Kokkos dataset validation?
- Hypothesis: ConTree sessions provide the required persistent filesystem semantics, while
  branching a prepared image can support concurrent Code RL grading.
- Experiment design: (1) adapter file/command smoke test, (2) known-answer Code RL grader
  invocation, (3) one minimal RL optimizer step, and (4) one existing CPU Kokkos candidate's
  baseline/test-only/gold validation transition.
- Success criteria: all interface operations succeed, the known-answer program receives reward,
  the RL run writes finite metrics and a rollout transcript, and the Kokkos validation report has
  `passed=true`.
- Control: existing Modal remains the default and its unit tests must continue to pass.

## LiveCodeBench-CPP GPT-OSS 20B RL experiment (2026-08-17)

Research question: can the existing agentic Code RL loop improve GPT-OSS 20B on NVIDIA's
LiveCodeBench-CPP problems when reward comes from isolated C++17 compilation and private tests?

Hypothesis: GPT-OSS 20B has enough initial code ability to create mixed rewards within groups,
while one tool-assisted revision gives it a useful compile/test feedback signal for GRPO.

Experiment design:

1. Pin the v6 dataset revision and deterministically shuffle with seed 0.
2. Hold out 32 problems for evaluation and use only the complementary 422 for training.
3. Run a small end-to-end grader/evaluation smoke, then evaluate all 32 held-out problems at
   pass@1 before any optimizer step.
4. Start LoRA RL with the recommended renderer, group size 8, 32K generation limit, and periodic
   evaluation on the unchanged holdout.
5. Monitor reward variance, correctness, format compliance, rollout errors, compilation errors,
   and actual transcripts during the first steps.

Controls and success criteria: train/eval question IDs must be disjoint; Python DeepCoder tests
must remain unchanged; known-good stdin and functional C++ submissions must pass the grader;
the baseline must finish without infrastructure errors; and training must produce finite loss,
nonzero within-group reward variance, and at least one valid checkpoint. Results on a checkpoint
trained with this dataset must not be reported as an uncontaminated public LiveCodeBench score.
