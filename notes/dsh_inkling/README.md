# DSH as the harness for Inkling-Small on SWE-kokkos-bench

Branch `dsh-inkling-kokkos`. Goal: **raise Inkling-Small's SWE-kokkos-bench score by
iterating the harness**, with DSH (`dsh-headless`) as the harness under test and the
ConTree or Rivanna Apptainer sandbox as the execution environment.

## Result

**DSH with the grader's build target stated is the only arm above parity: 50.8% over five draws,
against a DSH control mean of 46.8% and a native mean of 46.7% over three draws. The
improvement is consistent at about +4 points and does not reach significance, so the objective
is not met.**

| arm | draws | solved | mean | test-target build rate |
| --- | --- | ---: | ---: | ---: |
| native loop | 3 | 50, 50, 40 | 46.7% | — |
| DSH control | 5 | 55, 46, 47, 43, 43 | 46.8% | 5% |
| DSH + coverage nudge | 1 | 49 | 49.0% | — |
| DSH + re-arm & echo | 1 | 44 | 44.0% | 16% |
| **DSH + grader target stated** | **5** | **54, 51, 49, 50, 50** | **50.8%** | **73%** |

Paired against the DSH control: +4.0 points, 24 won against 15 lost, p = 0.200. Against the
native loop: +4.1, 25/17, p = 0.280. Unpaired on arm totals, t = 1.69 and 1.20.

Two findings stand on their own:

- **Parity between the unmodified harnesses** is now measured on eight draws: DSH control 46.8
  against native 46.7, a 0.13-point gap.
- **The mechanism works.** A controlled experiment showed the model builds the library while
  grading compiles a unit-test target, so it never saw its own compile errors. Naming the
  target moved test-target building from 5% to 73%, reproducing in all five draws, and it
  mediates the outcome (builders pass 59%, non-builders 33%). Freeing episodes from that
  failure mode moved the mean ~4 points and no further, because the freed episodes mostly
  failed a different way.

Three things are worth reading before reusing this work:

1. **A verifier bug in the driver** inflated every early DSH score. The published payload's
   `test.sh` selects zero tests and then awards reward 1, so an unmodified repository passed.
   Everything before `full20` is unusable; the fix and its guard test are in
   `harness/contree/run_task_contree.py` and `run_task_contree_test.py`.
2. **The noise floor is ±7 points.** The same configuration run twice produced 53 and 45, and
   the treatment's three draws spread 5 points. Effects of 1-6 points need 6-8 draws per arm;
   this project ran three. Three early "positive results" were retracted for this reason.
3. **Aggregate correlations reverse under task control.** Build count, patch size and build
   timing all looked predictive in aggregate and all reversed, or vanished, within tasks,
   because the same latent variable drives both. Only within-task comparisons survived: the
   coverage analysis (12 against 0, p = 0.0005) and the target-state mechanism.

Full detail, including every rejected variant and every corrected conclusion, is in
[`RESULTS.md`](RESULTS.md).

## Roles

| Component | Role |
| --- | --- |
| **DSH** (`@deepseek-ai/dsh`, v0.2.0-rc.2) | The agent harness being iterated: prompt, tool surface, context management, turn budgets |
| **Inkling-Small** (`thinkingmachines/Inkling-Small:peft:262144`) | The model under test, reached through Tinker's OpenAI-compatible endpoint |
| **SWE-kokkos-bench** (`luosuu/SWE-kokkos-bench`, local v2 payload, 100 tasks) | The benchmark and its hidden-verifier grading |
| **Rivanna + Apptainer** | Per-task containerised execution; `tinker_cookbook/sandbox/apptainer_sandbox.py` already implements the adapter |

DSH reaches a non-DeepSeek model through `dsh-llm-pi-ai`, which accepts custom
OpenAI-compatible routes (`baseURL` + `apiKeyEnv` + a hand-declared model catalog), so no
DSH fork is required to point it at Tinker. The existing desktop profile
(`~/.dsh/profiles/desktop/cordis.patch.yml`) shows the pattern for a custom `baseURL`.

## Prior work on Rivanna (inventory, 2026-10-05)

Before building anything, note that **most of this pipeline already exists** on Rivanna
under `/scratch/fad3ew/openhands-smoke/`. Do not rebuild it.

### A working DSH-on-Rivanna integration

`harness-cross-eval-20260924/` contains a complete, previously-run integration:

- `dsh-runtime/deepseek.patch.yml` — a DSH profile already pointed at
  `thinkingmachines/Inkling-Small:peft:262144` over Tinker's OpenAI-compatible endpoint
  (`api: openai-completions`, `baseURL: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`,
  `compat: {supportsDeveloperRole: false, maxTokensField: max_tokens}`). It disables
  `tool-bash`, `tool-fs`, `tool-fs-search`, `tool-web`, `tool-subagent*`, `tool-workflow`,
  `tool-goal`, `tool-todo`, `tool-skill`, etc., leaving exactly **one** repository tool.
- `dsh-runtime/deepseek-bridge.mjs` — that tool: a `bash` plugin that talks over a Unix
  socket to `SandboxInterface.run_command`. The model runs on the host (for credentials);
  **all repository access stays inside the network-isolated Apptainer sandbox**.
- `eval_deepseek.py` — the driver: validates the SIF hash against a recorded validated
  snapshot, asserts NOP=0/oracle=1, checks hidden assets are absent from the sandbox,
  serves the socket, runs `dsh --profile headless --patch ...`, captures the candidate,
  and grades with the Harbor verifier.
- `eval.sbatch` — `module load apptainer/1.4.5 nodejs/24.5.0`, array job. Pinned
  `@deepseek-ai/dsh@0.1.5-rc.3` (the local DSH is newer, `0.2.0-rc.2`).

This is the correct architecture to extend; it already solves the model-credential,
sandbox-isolation, candidate-capture and grading problems.

### Validated task images (no SIF translation needed)

- `pi-skill-eval-20260922/tasks/<task>/environment/sif.path` points at
  `inkling-eval/images/<task>.sif`; `inkling-eval/images/` holds **104 SIFs (~116 GB)**.
- `v24-release-20261002/images/` holds **99 SIFs (~101 GB)** with a 100-task
  `runtime-manifest.json` (per-task source/runtime Dockerfile digests, base SIF, and
  `"translation": "unchanged task commands; compile parallelism 4"`), plus
  `release-gate.json` reporting `ready: true, accepted: 100/100` with NOP/oracle records.

So the answer to "should we translate Dockerfiles to SIFs?" is **no — it is already
done and validated**; reuse `v24-release-20261002`.

### A same-model leaderboard baseline (the number to beat)

`v24-leaderboard-full-20261003/` is a completed, audited 400-trial run
(`progress.json: publishable: true`, 100 tasks × 4 models, 100/100 valid each, 0 errors)
using a **native Chat Completions tool loop**, 40 turns / 80 tools / 65,536 output tokens,
pass@1, `reasoning_effort 0.9` + `separate_reasoning: true`:

| Model | Solved |
| --- | ---: |
| `thinkingmachines/Inkling-Small:peft:262144` | **52/100 (52%)** |
| `thinkingmachines/Inkling:peft:262144` | 43/100 (43%) |
| `zai-org/GLM-5.3:peft:262144` | 56/100 (56%) |
| `Qwen/Qwen3.8-27B:peft:262144` | 45/100 (45%) |

**Inkling-Small's harness-independent native baseline is 52%**, not the 50.0% in the
self-training corpus. Use the 52% row as the comparison target, and budget-match any DSH
number to 40 turns / 80 tools / 65,536 output tokens.

### Cross-harness comparison already protocolled

`harness-engineering-closeout-20261001/harness-compare-protocol/` defines arms
(`native_sampler`, `native_chat`, `pi_legacy`, `pi_protocol`, `pi_protocol_prompt`,
`claude_legacy`, `claude_budget_control`) with a frozen design (`seed`, `max_attempts`,
`task_digest`, `code_sha256`, `comparability`). `FULL100_REPORT.md` records a Claude Code
100-task pass@4 run (25/100 pass@4, 47/400 pass@1) whose integration was diagnosed as
unhealthy (all 400 sessions ended on thinking-only or unexecuted MCP-call syntax — see
`TERMINATION_DIAGNOSIS.md`). Its known comparability gap is exactly DSH's: **the stock
pi-ai provider did not expose Tinker's `reasoning_effort` / `separate_reasoning`, so model
identity matched but inference parameters did not.**

## Environment facts established

Verified against `fad3ew@login.hpc.virginia.edu` on 2026-10-05:

- **Access**: the operator's ControlMaster session works, and an ed25519 key
  (`~/.ssh/id_ed25519`) has been installed into the remote `authorized_keys`, so
  `ssh -i ~/.ssh/id_ed25519 fad3ew@login.hpc.virginia.edu` now authenticates
  non-interactively (**verified**). Prefer this over the multiplexed socket, which expires
  when the operator's terminal closes.
- **Slurm** is not on the default `PATH`: `/opt/slurm/25.05.6_mam/bin`. Partitions
  include `interactive`, `standard`, `gpu`, `bii`, `bii-gpu`, `bii-largemem`. The
  operator's queue is heavily loaded, so any work here competes for quota.
- **Apptainer**: `module load apptainer/1.5.0` is the default but **`apptainer pull`
  fails** with `proot error: ptrace(TRACEME): Operation not permitted` on both the login
  node and compute nodes. **`apptainer/1.4.5` pulls successfully** (and is what the prior
  sbatch scripts already use). `exec` works on both.
- **No Docker/podman/buildah on Rivanna.** `apptainer build` does **not** accept a
  Dockerfile directly (`invalid header keyword found: from`); a `Bootstrap: docker` def
  file works (verified: built an Ubuntu 24.04 SIF on a compute node). This only matters if
  a new image is ever needed — the validated SIFs already exist.
- **Offline mode prerequisites hold**: `unshare --user --map-root-user --net` works inside
  and outside the container, `--fakeroot` works, `/dev/fuse` and loop devices exist. So
  the adapter's `allow_network=False` path is viable here.
- **Node**: `module load nodejs/24.5.0` is available for the DSH runtime.
- The login node **can** reach container registries, `github.com`, and the Tinker endpoint.
- Task payloads carry `environment/sif.path` pointers, which is exactly what
  `apptainer_sandbox_factory` reads.
- `TINKER_API_KEY` lives in `/scratch/fad3ew/openhands-smoke/inkling-eval.env` (mode 600);
  the prior driver loads it on the host only and never passes it into the sandbox.

## Analysis baseline (done)

[`trajectory_failure_modes.md`](trajectory_failure_modes.md) analyses the frozen 400-episode
Inkling-Small corpus (score 50.0%). The operative findings:

- 63.7% of episodes end at the 40-turn cap, scoring 34.9%, versus 85.4% when the model
  finishes on its own.
- **80/400 episodes produce an empty patch and score 0.0%** — a clean 20-point loss.
- Repetition is not the problem (mean repeated-command share 0.025).
- Capped episodes first edit the repo at turn 21.8 vs 9.5 for completed ones; 44/80
  empty-patch episodes never wrote anywhere, and of the writes that happened, 64 went to
  `/tmp` versus 1 to the repo.

So the first harness hypothesis is **"the agent explores and never commits an edit"**, and
the first knobs are write-forcing, scratch-probe cost, and build-loop cost.

## Plan

Each step is gated on the previous one's evidence. Harness edits are the only intended
variable; the model, task payload, grader and turn accounting stay fixed within a
comparison. Steps 0 and 1 are largely **already done** by the prior work above, so the
plan starts by *reusing and upgrading* rather than building.

### Step 0 — adopt the existing integration (mostly done)

- Reuse `harness-cross-eval-20260924/`: profile, socket bridge, driver, sbatch.
- **Upgrade the DSH version** from pinned `0.1.5-rc.3` to a current line, and check whether
  the newer `dsh-llm-pi-ai` can express Tinker's `reasoning_effort` /
  `separate_reasoning`. The 0.2.0-rc.2 README documents `reasoningEfforts`,
  `compat` (including how a thinking level travels) and `thinkingTokenBudgetField`
  ("selects the reasoning-budget parameter" for self-hosted Chat Completions), none of
  which the 0.1.5 profile used. **This is the main comparability fix** and it is on the
  critical path: without it a DSH score is not comparable to the 52% native baseline.
- Reuse `/scratch/fad3ew/openhands-smoke/inkling-eval.env` for `TINKER_API_KEY` on the
  host only.

### Step 1 — execution path (done; verify, do not rebuild)

- Reuse the **validated** `v24-release-20261002` SIFs and their `release-gate.json`
  (`ready: true`, 100/100 accepted, NOP=0/oracle=1). Do **not** re-translate Dockerfiles.
- Reuse task payloads with `environment/sif.path`; `apptainer_sandbox_factory` reads them
  directly.
- Pin `apptainer/1.4.5` and `nodejs/24.5.0` in sbatch, as the prior scripts do.
- Spot-check that the recorded SIF hashes still match before any run (the prior driver
  already refuses to proceed otherwise — keep that check).

### Step 2 — baseline and harness loop

- Re-run the DSH arm to establish its pass@1 on the same 100 tasks, budget-matched to
  40 turns / 80 tools / 65,536 output tokens, and compare against **52%** (Inkling-Small
  native Chat Completions), not 50%.
- Iterate one harness change at a time. The failure-mode analysis below nominates the
  first candidates: a write-forcing mechanism, scratch-probe cost, and build-loop cost.
- Reuse the frozen `harness-compare-protocol` design (seed, task digest, slots) so arms
  stay comparable; register DSH as an arm rather than inventing a new protocol.
- Report pass@1 and per-task deltas; separate always-fail tasks from mixed ones rather
  than averaging them together.

### Step 3 — analysis of DSH trajectories

- The prior driver writes `events.jsonl` (every bash call and result) plus the DSH
  session log, so a DSH corpus is already analysable.
- Normalize DSH events into the corpus shape `analyze_trajectories.py` consumes, then
  compare: did the empty-patch share fall, and did the mean turn of first repository write
  drop?

## Guardrails

- **Answer-lookup must not grow.** The environment is network-disabled (`allow_network=False`)
  and the exported image is stripped to the base commit with remotes/refs/reflogs removed;
  the verifier pins the commit SHA and ignores Git replacement refs. The prior driver
  additionally asserts `/tests` and `/solution` are absent from the sandbox — keep that.
- **Do not pass the model credential into the sandbox.** The prior architecture keeps DSH
  and `TINKER_API_KEY` on the host and exposes only a socket bridge.
- **Do not mix harness arms or sandbox backends** within a comparison.
- **Never retry an incorrect solution as an infrastructure failure.**
- Keep credentials in the environment or `.env`, never in artifacts.
- **Budget-match before claiming a win.** The original DSH pilot ran 75 model turns
  against a 40-turn reference and 80 tool calls; a solution found after the reference cap
  is an exploratory success, not a budget-matched improvement.

## Open questions

1. **Reasoning parameters.** Can current DSH express Tinker's numeric `reasoning_effort:
   0.9` with `separate_reasoning: true`? This is the known comparability gap; resolve it
   before trusting any DSH score against the 52% row.
2. **Tool surface.** The prior profile already reduces DSH to a single `bash` tool, which
   matches the baseline harness — good for attribution. Keep it fixed while iterating
   harness behavior, and treat tool-surface changes as their own arm.
3. **Slurm capacity.** The operator's queue is already deep. The prior full runs used
   `bii-largemem`/CPU partitions at 4-8 concurrency and ~2 h jobs; confirm an allocation
   policy before launching anything wide.
