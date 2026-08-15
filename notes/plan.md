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
