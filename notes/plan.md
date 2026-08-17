# Kokkos Coding-RL Dataset: Phase 0 Plan

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
