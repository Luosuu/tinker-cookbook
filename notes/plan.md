# Kokkos Coding-RL Dataset: Phase 0 Plan

## Hosted Chat Completions integration smoke (2026-09-06)

Question: can the existing annotation, instruction-audit and Harbor evaluation paths use
the hosted API without changing sandbox validation, mixing task histories or losing usage?
Add an opt-in transport; keep native token collection and the running self-training
comparison unchanged. First run offline protocol and regression tests. After committing
this plan and implementation, make one real instruction-audit request and run one
independent Small evaluation on the already Oracle/NOP-validated task `kokkos__kokkos-6375`.
Use effort 0.9, temperature 1, at most 40 turns and 65,536 output tokens, with no full-task
retry. Keep evidence in a separate ignored output directory. This tests integration only;
one task cannot establish transport parity or a benchmark improvement. No dataset edits,
new collection batches or optimizer steps are part of this smoke.

## Hosted OpenAI-compatible endpoint connectivity (2026-09-06)

Test the hosted Tinker Chat Completions endpoint with the existing Tinker API key.
Use short, synthetic prompts to verify text completion, a function-call/result roundtrip,
streaming, usage reporting, and explicit Inkling effort 0.9. Preserve request configuration
and response evidence locally. The smoke script makes at most four short generations per
model and does not start a benchmark or change weights. Check base-model and existing
sampler-checkpoint identifiers separately where available. Keep this connectivity test
independent of the fixed 40-turn research experiment and its native-token collection path.

## Fixed 40-turn successful-trajectory self-training (2026-09-06)

Question: can self-training improve Inkling-Small's Kokkos repair success and reduce
the turns and sampled tokens needed per delivered repair, under a hard 40-turn cap?
The cap stays fixed for collection, the base model, and both trained checkpoints.

Compare three conditions: the unchanged base model, one randomly selected qualified
success per training task, and the shortest qualified success per same training task
(fewest turns, then sampled tokens). The two training arms use identical task coverage,
one demonstration per task, equal total loss weight per trajectory, and the same optimizer
settings. This tests demonstration selection; it does not establish a general optimal RL
algorithm. Short demonstrations are the explicit efficiency training intervention.

Use the current prepared 100-task payload snapshot, sorted then shuffled with seed 7,
with 80 training tasks and 20 disjoint evaluation tasks. Do not use historical trajectories
or heldout outcomes to select demonstrations. Save task digests and the committed code
revision before starting remote work. A small fixed validation subset precedes the full run.

Required gates, in order:

1. In fresh network-disabled sandboxes, verify that the baseline is the only reachable
   commit, the merge commit is absent, and hidden test/solution directories are absent.
   Require NOP=0 and packaged Oracle=1 for every current task snapshot.
2. Evaluate the unchanged Small model on the 20 heldout tasks with four samples per task.
   Resolve infrastructure errors before collection or training; retain failed attempts
   separately and never retry an incorrect solution as an infrastructure failure.
3. Collect four new Small trajectories on each of the 80 training tasks. Candidate donors
   must pass, terminate naturally within 40 turns, have a nonempty source patch, and have
   no parse-error, answer-lookup, or hidden-material-access audit flag. Cap-terminated
   successes count in evaluation but are excluded from both training arms.
4. Reapply each candidate patch in a fresh sandbox and require a second reward of one.
   Require at least 16 distinct qualified training tasks; otherwise stop for diagnosis.
   Preserve exact sampled tokens, including thinking-effort conditioning, and supervise
   only action tokens. Validate every token/target/mask and inspect decoded examples.
5. Start each SFT arm independently from `thinkingmachines/Inkling-Small:peft:262144`,
   LoRA rank 32, constant LR 1e-5, one epoch, four tasks per batch (at most 20 updates).
   Evaluate both final checkpoints on the identical heldout task/sample slots.

Shared rollout settings: automatically recommended renderer, explicit thinking effort 0.9,
temperature 1.0, 40 turns, 80 tool calls, 16,384 tokens per response, 65,536 sampled tokens,
114,688 trajectory tokens, and a 900-second command/grader timeout. Sandbox concurrency
is four, with ConTree runtime CMake builds limited to one compiler process per sandbox;
no sampling timeouts or sampling retry wrappers are added. Retain verifier stdout/stderr
for every validation, candidate regrade, and evaluated rollout. The initial budget
is 200 verifier checks, 80 baseline rollouts, 320 collection rollouts, fresh regrades only
for qualified candidates, and 160 checkpoint-evaluation rollouts. No automatic sweep or
budget expansion is permitted by this recipe.

Report pass@1, infrastructure failures, mean turns on all trials and on successes,
sampled tokens, and total turns/tokens per delivered success (including incorrect trials).
Also report observed successes that ended by turns 20/30/40; these are not independent
lower-cap evaluations. Use task-clustered paired uncertainty for comparisons. A promising
efficiency result requires lower cost per delivered success without a material observed
success-rate drop; report uncertainty explicitly rather than claiming noninferiority from
only 20 tasks. Before broader training, inspect failures and both arms' learning curves.

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

## Nebius Kokkos pass@1 comparison (2026-09-06)

Compare the four requested hosted models on the same frozen 100-task snapshot,
using one attempt per task, at most 40 turns, 65,536 output tokens in total,
16,384 output tokens per request, 80 tool calls and the existing verifier.
Use Nebius Chat Completions with explicit high reasoning effort, server-default
sampling temperature, and preserve assistant reasoning across tool turns.
The authenticated verbose model catalog supplies exact model IDs and token prices;
cache discounts are not published in that response, so cost estimates use the
undiscounted input price and are not invoices.

Run one already validated task for each model first; these four attempts count
within the final 400 model-task attempts. After all four complete without API or
sandbox errors, continue only tasks whose NOP/Oracle gate has passed. The four
models share a single concurrency limit of four new sandboxes while the existing
self-training verifier process continues. Failed or pending verifier tasks wait;
they are not counted as model failures. Preserve each model-task transcript,
grading evidence, task digest and token/cost/turn measurements. A restart never
resamples a completed or errored trajectory. Diagnose errors before deciding
whether an infrastructure-only recovery can preserve the pass@1 protocol.

Success means 100 valid attempts per model and comparable pass@1, turn counts,
token counts, and estimated inference cost. Partial valid-task pass rates are
labeled partial and never substituted for the final 100-task denominator.

Two tasks have confirmed environment failures under the default ConTree resources:
7244 requires a single 4 GiB allocation and 8164 exceeds the cold-build deadline.
Use a fixed resource policy for all four models: those two tasks run on Modal
with 16 GiB memory and four CPUs; the remaining tasks use ConTree. The resource
mapping is part of every model's resumable experiment identity. Before either
exception task is sampled, its unchanged verifier must pass independent NOP/Oracle
checks on that backend. Keep these checks in the sweep's validation_overrides
folder and preserve the running self-training experiment's original evidence.

## Resource validation follow-up (2026-09-06)

The fixed 16 GiB Modal gate passed task 7244. Task 8164 still timed out during
single-thread compilation (129/141 units), before CTest. Independently validate
its unchanged task snapshot with Modal 16 GiB/four CPUs and explicit build
parallelism four, retaining the 900-second verifier timeout. No model sampling
is part of this diagnostic. Preserve the previous failed evidence; write only
to resource_validation_v2, not the running controller's validation_overrides.
The current processes keep their existing policies. A later evaluation on the
repaired task requires a separate identity or a documented drained-controller
transition, with the same policy applied to all four models.

Task 8164 passed this four-thread Modal gate. Investigate the subsequent Oracle
timeouts on 8399 and 8827 with the same independent NOP/Oracle diagnostic,
one task at a time, in resource_validation_v3. Reuse the unchanged frozen task
bytes, keep the 900-second deadline, and make no model inference requests.
These records also cannot unlock the running controller's older resource policy.

## Instruction quality gate follow-up (2026-09-06)

Task 7043's frozen instruction asks to suppress anonymous-namespace prefixes,
but its gold implementation and compiler-specific hidden assertions preserve
them. A passing NOP/Oracle pair therefore does not establish instruction
correctness. Keep the running 100-task evaluation snapshot and its raw scores
unchanged, flag this task in the report, and do not resample model failures.
The reviewed instruction override now requires preserving those prefixes.
Self-training must reject the known contradictory wording before any baseline
or donor sampling. Recovery requires applying the override to a fresh dataset
and experiment snapshot, recording the new hashes, and rerunning its gate;
never edit the active manifest or borrow a validation record with another hash.
The original validation process may finish collecting its remaining evidence,
but its existing resource failures already prevent it advancing to sampling.

## Bounded provider-outage recovery (2026-09-06)

The exact DeepSeek-V4-Flash-0731 endpoint temporarily disappeared from the
authenticated catalog and rejected two requests with HTTP 404. The catalog and
normal inference subsequently recovered. Preserve those original error records;
never substitute another model or resample an already generated prefix.

After the original controller releases its shared scheduling lock, permit one
separate recovery per affected task. Task 7074 produced no output and may make
its first valid sample. Task 7043 has exactly one complete turn containing two
reviewed read-only shell commands. Return that original response from a local
transport, repeat only those commands on the identical task snapshot, and require
byte-for-byte identical observations before sending the next live model request.
Any command, request, task-hash or observation mismatch blocks continuation.
The recorded first turn still consumes the original trajectory's turn/token
budget; no fresh first-turn sampling is allowed. Keep the contradictory 7043
instruction and raw grading policy unchanged for this frozen evaluation.

Bind the recovery to the original model/config identity and full task manifest,
keep an exclusive attempt marker and source-response provenance, and retain
partial responses if interrupted. Record logical trajectory usage separately
from newly billable usage: subtract cached-prefix usage only when it was actually
replayed. Aggregate spend as original usage plus incremental recovery usage,
never as the sum of both full trajectory totals. Offline transport/harness tests
must prove no first-response resampling, no live requests after differing tool
observations, no negative accounting before replay, and exclusion while the
original controller owns the shared slots. Commit before starting recovery.

The remaining previously unsampled resource-exception tasks use a separate
Modal resource phase after the same controller lock becomes available. Admit
only explicit task names with exact matching NOP/Oracle evidence for 16 GiB,
four CPUs, build parallelism four and the unchanged 900-second deadline. Reuse
each original model's other settings and prices, but record the new resource
policy in a distinct evaluation identity. Refuse any task with an original
attempt marker or result, cap aggregate concurrency at four, and never resample
completed, errored or interrupted phase attempts. Preserve phase-level metrics
separately; the final full-dataset report joins unique model-task results across
phases and continues to use the original 100-task denominator.

The future recovery/resource phases also retain a candidate patch before hidden
tests are injected. Export against the verifier's immutable baseline, including
agent-committed changes and untracked source files; save the original sandbox ID,
baseline and HEAD commits, byte count and SHA-256. This is a read-only audit hook,
does not consume another model turn, and records failure explicitly without
changing the grade. Keep it optional and outside CLI configuration so the running
controller and its existing resume identities are unaffected. Previously
completed attempts lack this artifact; do not claim they can all be regraded
without reconstructing their source changes from the retained evidence.

## Runtime-test coverage repair (2026-09-06)

Oracle logs revealed commands that exited successfully after selecting zero
tests. The annotation normalizer previously expanded TEST_CATEGORY only for
runtime F2P tasks, leaving runtime P2P checks on compile-failure tasks unchanged.
Use one selector normalizer in annotation, local/sandbox validation and Harbor
export: preserve every positive/negative filter alternative, repair literal
macro suites and quoting, and apply a renamed case only when the hidden patch
contains unambiguous corresponding old/new declarations. Baseline P2P uses the
old case name; test-only and gold stages use the post-patch name. Do not remove
an invalid runtime P2P check to make an annotation validate.

Use the same standalone runtime guard in all verification paths. GoogleTest
must report nonzero selected tests and, for a successful command, matching
nonzero passed summaries. Reject any zero-test invocation, including one mixed
with successful invocations, all-skipped runs and missing execution evidence.
CTest runs verbosely from the build directory so nested empty GoogleTest runs
are visible; reject missing/empty CTest execution and masked failures. A coverage
error cannot count as an expected F2P failure. Compile-only checks remain legal.
These checks use output evidence compatible with older bundled toolchains,
rather than assuming a recently added GoogleTest flag is available.

Audit all 100 frozen command sets and create a distinct corrected task snapshot;
never rewrite the running task payload. Unknown selectors and hardware-specific
requirements remain blocked for diagnosis. Revalidate under the corrected hash
before sampling, and regrade retained candidates without resampling model text.
The original quarantined evidence and raw scores stay available for comparison.

## Runtime coverage audit (2026-09-06)

Oracle logs exposed zero-test GoogleTest invocations on 18 tasks, including
14 whose visible invocations all ran zero tests. Several added runtime
assertions were therefore not exercised despite NOP/Oracle separation.
Literal TEST_CATEGORY filters and renamed test cases are confirmed causes.
Quarantine their original validation records, preserving bytes and hashes,
to remove eligibility for further sampling without stopping active rollouts
or changing either frozen task snapshot. Watch subsequent Oracle records for
the same symptom. Review compile-only tasks separately before deciding whether
an empty runtime selection is intentional; do not equate every zero-test log
with a missing compile-time check.

Fix selection and add explicit runtime coverage checks in a fresh dataset
version before resuming affected tasks. Preserve existing raw scores with a
coverage flag, and regrade recoverable candidate patches without new sampling.
The resource and provider-error recovery phases remain unstarted until this
review establishes their valid grading policy. The original 100-task evaluation
is incomplete while any task lacks trustworthy coverage.

Task 9147's Oracle also fails because CudaInterOpGraph cannot load libcuda.so.1.
Treat this as a missing GPU runtime gate, not a model failure; additional CPU
memory alone does not repair it.

## Preserve remaining candidate patches (2026-09-07)

Temporarily hold all remaining eligible validation records, preserving their
original bytes and locations in a transition ledger. Continue holding newly
completed gates so the four active evaluations finish naturally with no new
dispatch. Once the controller reports zero active tasks, preserve its launch
and status evidence and stop only that idle controller. Resume the same frozen
configuration with the existing read-only candidate artifact hook enabled for
each remaining task. Restore eligible records only after rechecking the known
zero-test quarantine rules; compromised gates stay quarantined. This adds
patch retention without changing model sampling, task hashes, result identities
or the fixed 40-turn budget, and never resamples completed attempts.

## Regrade retained candidates after transport failures (2026-09-07)

The GLM 7441 and DeepSeek Flash 7485 trajectories completed generation but
their grader operation polling failed with HTTP read timeouts. Both have a
complete candidate patch captured before hidden tests. Regrade each once,
serially, in a fresh ConTree sandbox using its original frozen task hash,
build parallelism one and the 900-second grader budget. Verify the candidate
byte count, SHA-256 and base commit against the original task before applying.
Make no model requests, keep the original error/results unchanged, and record
separate patch-only evidence whether the answer passes or fails. Do not retry
a negative grading result. This uses one additional sandbox slot after the
original main validation process has exited.

Task 9147 already declares one L4 GPU, four CPUs and 16 GiB RAM in its original
task configuration. Validate its corrected v6 verifier separately with those
explicit Modal resources, build parallelism four, network disabled, and the
unchanged 900-second grader limit. Run NOP then Oracle once, with no model
requests; record the frozen task hash and GPU policy in the evidence. This
policy is isolated from the running CPU evaluation and cannot unlock its
original resource identity. The sandbox factory's optional GPU defaults to
None and only applies to the explicitly selected tasks.

## Network policy for future evaluation phases (2026-09-07)

Three pending inference connections remain bound to an old VPN source address.
A read-only probe bound to that address cannot connect, while the new address
receives a response immediately. Add an explicit future transport policy with
TCP keepalive, no generation read deadline, and zero HTTP/SDK automatic retries.
Record the policy in a new phase identity; it must not silently change the
running experiment. A response lost after sending a request has unknown usage
and requires explicit recovery review, rather than assuming zero cost or
silently drawing another answer. Offline tests verify that one read failure
causes exactly one request and leaves the generation read timeout unlimited.
This helper cannot repair existing sockets. Preserve current pending requests
while investigating whether their last successful sandbox state can be
recovered through read-only immutable-image metadata and files.

## Review exact runtime selectors (2026-09-07)

The cached original source confirms three selectors need explicit corrections
beyond syntax normalization: 7428 uses the executable name as a GoogleTest
suite, 8594 uses the wrong case for ScatterView tests, and 8967 invents a death
test suite name. Pin each reviewed replacement to the instance and base commit;
record source paths and SHA-256 evidence. Apply the same replacements in export,
local validation, sandbox validation, and annotation. Keep unrelated selectors
and revisions unchanged, and retain the runtime coverage guard.

Prepare a second corrected snapshot with these replacements, preserving the
first unvalidated snapshot and original experiment payload. Before model calls,
validate exact new hashes with NOP and Oracle. Initial coverage checks exercise
an unchanged control (6375), build-stage runtime coverage (7040), renamed tests
(7088), and CTest (7074), using one additional sandbox at a time. Then validate
the three manually reviewed selectors. No recovery sampling or regrading claims
are permitted without the corresponding corrected verifier evidence.

The v2 control validator uses its own process lock and output directory and
waits until the original validator process has actually exited before opening
any sandbox. It validates the four initial cases serially and refuses to replay
an existing attempt. Waiting consumes no model calls or sandbox slots; the
ongoing Nebius worker retains its four-slot limit. Record the validator script
hash, repository commit, fixed snapshot hashes, and NOP/Oracle evidence.

Task kokkos-kernels-2864 has a second reviewed command correction: its configured
build does not expose the sparse/unit_test/test build target. The cached base
image's generated CTest file confirms four tests in that same directory. Use
CTest directly for that directory, pinned to the exact original command and
base commit, preserving the test scope and enforcing nonempty nested runtime
coverage. Prepare this change as snapshot v3; v2's four control gates continue
against their frozen hashes. Validate 2864 separately before making any model
attempt eligible.

## Repair reviewed target and test-role annotations (2026-09-07)

Seven further quarantined tasks need source-backed annotation repairs. Build
Serial2 for ViewAPI_b (7308) and ViewCopy_a (9055); select the registered serial
containers suite (7517), constructor-property test family (7605), and existing
C-style finalize/free check (7675). Route 8838's new core legacy-layout test to
ViewSupport and classify it as F2P, retaining the existing containers layout
check as P2P; that F2P classification must pass validation before use. For 9027,
run the existing serial.graph_then_tag once in its actual Serial1 executable.

Pin these changes to the original instance, base commit, hidden test patch
hash, and original annotation fields. Reject changed annotations rather than
silently applying an outdated review. Apply the repaired build targets and
command roles consistently to export and both validators. Preserve prior
snapshots and prepare v4 with only these seven task changes. No model sampling
or successful-gate claims are authorized until exact new hashes pass NOP/Oracle.

The initial four v2 control validations all ended in ConTree ApiTimeoutError
without a verdict, after the original validator exited. Retain those failures
and permit one explicit infrastructure retry in a separate directory, using
identical task hashes, one sandbox, no model calls, and the unchanged SDK HTTP
timeout. Record the prior failure, failed phase, endpoint, and timeout type.
A fresh read-only token-info probe succeeded before this retry. Any further
failure remains blocked for diagnosis; there is no automatic retry loop.

## Complete backend and scalar-type coverage repairs (2026-09-07)

Generated CTest files confirm that kernels-2935/3130/3138 register tests as
batched_dla_openmp or batched_dla_serial, without the KokkosKernels_ executable
prefix. Repair these exact commands at their original revisions. Task 9260's
source defines the suite as lowercase openmp. Kernels-3217's complex-float norm
test is correctly named but excluded by the original CMake configuration;
enable INST_COMPLEX_FLOAT rather than substituting a double-only test. Preserve
the baseline CMake-cache hash proving that the option was OFF.

Prepare snapshot v5 separately. Regenerate kernels-3217's Dockerfile because
its scalar configuration changes; verifier commands alone cannot enable a test
that was never compiled. Keep the remaining payloads unchanged except for their
reviewed commands and metadata. Validate exact hashes before sampling; compiler
cost or resource failures remain explicit infrastructure/coverage blockers.

Two final framework-name corrections are reviewed from the original source:
7458's CTest name contains a misplaced underscore, and 8819's isinf check is a
GoogleTest case inside Serial1 rather than a CTest test. Preserve the exact case
and executable scope in snapshot v6. Task 8891 additionally selects an FP16
GoogleTest check that explicitly skips without native 16-bit support; keep it
blocked for a justified backend or annotation correction. An all-skipped test
cannot qualify as runtime coverage.

Task 8891 is explicitly a CPU task. Its existing
mathematical_functions_floating_point_manipulation_functions test exercises
nextafter with supported ordinary scalar types, so replace the incorrectly
selected FP16-only P2P case with that existing CPU test. Pin the replacement to
the original base, hidden-test hash, and P2P command. Preserve the new nexttoward
hidden regression, F2P build stage, production patch, and CPU resources. Prepare
v7 without changing v6; independently validate the corrected case before use.
Task 9147 retains its originally requested L4 GPU under a separate resource gate.

## Validate the corrected full snapshot (2026-09-07)

Use the frozen v7 snapshot with a resumable, zero-inference verifier runner.
Reuse only successful NOP/Oracle evidence with exactly the same task hash,
backend, GPU, CPU, memory, build parallelism, and grader timeout. Preserve and
block prior failures and interrupted attempts rather than silently trying again.
Record immutable identity, launch history, task attempt markers, raw grading
logs, and provenance for reused evidence.

The static resource mapping is ConTree/build1 by default; Modal16GiB/4CPU/build1
for 7244; Modal16GiB/4CPU/build4 for 8164, 8399, and 8827; and Modal L4/16GiB/4CPU/
build4 for 8989 and 9147, matching their declared GPU requirements. All graders
retain 900 seconds. Undeclared GPU routing is rejected before any sandbox call.
Wait until all four control gates pass and the control, patch-only recovery,
and independent GPU-validation processes exit before launching the bulk sweep.
At most three new validation sandboxes run together, alongside the existing
four model slots. Do not repeat same-hash control or GPU gates already proven
under that exact resource policy.

The fourth control, 7074, now runs 128 real tests. Its Oracle passes 124 and
skips three, but fails view_allocation_large_rank because its 4 GiB allocation
exceeds ConTree's measured VM capacity. Preserve that NOP0/Oracle0 result; the
conditional bulk launcher stopped without dispatching any validation. Keep the
same frozen v7 payload and all tests, and explicitly move 7074 to Modal16GiB/
4CPU/build4. Validate its NOP/Oracle pair once in a separate resource directory,
with no inference. A failure under this new policy remains blocked. Once this
fourth control passes, reuse the three ConTree controls and the same-hash L4
9147 gate, and launch the remaining v7 gates with at most three shared slots.
Do not reuse the old low-memory result as evidence for the new policy or erase
it from the experiment record.

Full v7 validation exposed a further incorrect duplicate runtime command in
7089. Its Oracle completes both builds and passes the real mdspan_atomic_accessor
case in Serial1, then requests that same case from Serial2, which contains no
matching test. The hidden CMake patch adds this case only to TESTNAMES1B. Remove
only the second runtime command, pinning the original base, hidden-test hash,
and command list. Preserve both build targets, the F2P build stage, and the
real Serial1 regression. Prepare a new frozen snapshot without changing the
active v7 validation or its failed evidence. A later exact-hash NOP/Oracle gate
must qualify the repaired payload before any additional model sampling.

Validate only the repaired 7089 v8 payload once in a separate ConTree gate with
one sandbox and zero inference calls. The original four model slots and three
v7 validator slots leave one slot within the shared limit of eight. Keep all
other v8 tasks waiting; reuse same-hash v7 evidence later. Any failure of this
new gate remains terminal for diagnosis rather than triggering a retry.

The bulk Oracle for 8039 compiles successfully but its filter contains literal
C++ TEST declarations and selects zero cases. Source confirms Multi_streams
and Random_XorShift64 in the requested Random executable; the hidden patch
extends Multi_streams with the new asynchronous initialization regression.
Replace only those malformed selectors, pinning base, hidden patch, and original
commands. Keep the executable, both cases, and F2P build stage. Preserve v7/v8
and freeze v9 with this single additional repair. Validate only the new 8039
hash once with one ConTree sandbox, 900 seconds per grader, zero inference,
and total shared concurrency at most eight. A failed gate remains blocked.

## Require coverage of the changed behavior (2026-09-07)

NOP0/Oracle1 and nonzero P2P coverage do not establish that a new runtime
regression actually ran. A static audit found 21 candidate tasks whose added
TEST cases are not selected, including legitimate compile-only candidates and
partially covered test families. Review each candidate against its hidden patch,
registered executable, backend guards, and runtime assertions before deciding
the correction. Do not classify all candidates as failures automatically.

Examples requiring investigation include isnormal checked only by an existing
isnan case, fpclassify checked only by isinf, and nexttoward compiled but never
executed. Nexttoward additionally skips outside the default host execution
space, so its test must use the appropriate enabled host backend.

Keep the running snapshots and their NOP/Oracle evidence immutable. Prepare
explicit command/target corrections in a new snapshot and validate changed
hashes independently, reusing only unchanged matching evidence. Qualification
must require a completed behavior-coverage review tied to every final task
hash, with no unresolved blockers, before setting ready_for_sampling. Missing,
incomplete, or stale reviews leave sampling blocked even if all verifier pairs
pass. Keep the pair-pass count visible separately for progress monitoring.

The broader review also checks modified existing cases and the actual executable
behind CTest registrations. A build-only command can miss new numerical work
without any newly added TEST declaration, as with ScatterValue updates and the
batched rotm parameter layout. Preserve the original build/P2P checks and add
separate guarded invocations for the reviewed changed cases. An unrelated old
passing test must not hide a required case that selects zero tests, skips, or
fails. Local shell tests cover all four outcomes in a build-stage task.

Use the already enabled OpenMP executable for 7248 and 8891 because the required
cases explicitly skip outside the default execution/host space. Keep their
original Serial checks. Preserve the configured scalar scope in kernels: 3088
and 3089 instantiate double only; 3217 additionally enables complex<float> as
previously required. Execute every changed mode/norm family within that scope,
and explicitly record the unconfigured scalar/backend branches as uncovered.
For 7074 run the normal accessor and changed mdspan-conversion cases; its
inaccessible host/device-memory death cases explicitly skip on the declared
CPU backend and do not provide GPU coverage.

Bulk validation revealed two more framework/target mistakes after successful
full compilation: 8399 registers Timer in Serial2 but requested Serial1, and
8827 requested a GoogleTest case as a CTest registration. Correct only the
reviewed target/framework commands under pinned base, hidden-test, and original
annotation hashes. Keep the failed old-hash evidence immutable. Freeze the
combined corrected payload separately after source review; any new gate must
use its exact hash and static resource policy. Do not launch new gates during
the current ConTree transport-error burst, and do not retry old errors or begin
additional model sampling as a side effect of preparing the corrected snapshot.
