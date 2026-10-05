# DSH as the harness for Inkling-Small on SWE-kokkos-bench

Branch `dsh-inkling-kokkos`. Goal: **raise Inkling-Small's SWE-kokkos-bench score by
iterating the harness**, with DSH (`dsh-headless`) as the harness under test and the
Rivanna Apptainer sandbox as the execution environment.

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

## Environment facts established

Verified against `fad3ew@login.hpc.virginia.edu` on 2026-10-05:

- **Access**: reuse the operator's existing ControlMaster session
  (`ssh -o ControlPath=<socket>`); the login node is reached at `login.hpc.virginia.edu`
  (30.200.1.121). Key auth is not available non-interactively, so the multiplexed session
  is the only path and it must stay alive.
- **Slurm** is not on the default `PATH`: `/opt/slurm/25.05.6_mam/bin`. Partitions
  include `interactive`, `standard`, `gpu`, `bii`, `bii-gpu`, `bii-largemem`. The
  operator's queue is heavily loaded, so any work here competes for quota.
- **Apptainer**: `module load apptainer/1.5.0` is the default but **`apptainer pull`
  fails** with `proot error: ptrace(TRACEME): Operation not permitted` on both the login
  node and compute nodes. **`apptainer/1.4.5` pulls successfully.** `exec` works on both.
  Pin `1.4.5` for any image build.
- **Offline mode prerequisites hold**: `unshare --user --map-root-user --net` works inside
  and outside the container, `--fakeroot` works, `/dev/fuse` and loop devices exist. So
  the adapter's `allow_network=False` path is viable here.
- The login node **can** reach container registries and `github.com`; no proxy is needed.
- Not yet present on Rivanna: the current cookbook checkout, the v2 task payload, and
  `TINKER_API_KEY`.

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
comparison.

### Step 0 — reproducible local plumbing

- Freeze and record: DSH version, model id, task payload digest, grader revision.
- A DSH profile that routes to Tinker's OpenAI-compatible endpoint with Inkling-Small and
  a declared model catalog.
- Confirm `dsh --profile <p> "..."` runs a one-shot task locally against a trivial
  workspace before touching sandboxes.

### Step 1 — Rivanna execution path

- Pin `apptainer/1.4.5`; build one task SIF from the exported Dockerfile on a compute node
  (not the login node), with `APPTAINER_CACHEDIR` on scratch.
- Validate the SIF by importing and starting the OpenHands agent server, checking the task
  workdir, and running a no-change baseline (grader must score 0) plus the packaged Oracle
  (must score 1). **An infrastructure error is not a failed-test result.**
- Then wire `apptainer_sandbox_factory` into the evaluation driver. The adapter reads
  `environment/sif.path` or `environment/agent-server.sif`, so task payloads need one of
  those pointers.
- Keep concurrency inside one Slurm allocation and record the allocation's CPU/memory;
  the adapter grants visibility, not quotas.

### Step 2 — baseline and harness loop

- Measure the DSH baseline on a fixed task subset with a frozen split seed.
- Iterate one harness change at a time; each change is a single-variable comparison on the
  same tasks, with the same sandbox backend kept for the whole run (backend changes make
  results uninterpretable).
- Report pass@1 and per-task deltas, and separate the 32 always-fail tasks from the 37
  mixed ones rather than averaging them together.

### Step 3 — analysis of DSH trajectories

- Re-run `analyze_trajectories.py` against DSH-produced corpora to compare failure modes
  against this baseline.
- The comparison that matters: did the empty-patch share fall, and did the mean turn of
  first repository write drop?

## Guardrails

- **Answer-lookup must not grow.** The environment is network-disabled by default and the
  exported image is stripped to the base commit with remotes/refs/reflogs removed; the
  verifier pins the commit SHA and ignores Git replacement refs. Do not set
  `allow_network=True`.
- **Do not mix backends** within a comparison.
- **Never retry an incorrect solution as an infrastructure failure.**
- Keep credentials in the environment or `.env`, never in artifacts.
- Results are estimates from one corpus; a 20-point loss is worth attacking, but any
  claimed gain must survive the same task split.

## Open questions

1. Does Tinker's OpenAI-compatible endpoint render Inkling-Small identically to the native
   SDK path? The cookbook README warns server rendering can differ, so a paired transport
   comparison is needed before treating DSH numbers as comparable to the native baseline.
2. Which tool surface should DSH expose for this benchmark — DSH's native `bash`/`fs`
   tools, or a single bash tool matching the baseline harness? A different tool surface
   changes two things at once and breaks single-variable attribution.
3. Slurm capacity: the operator's queue is already deep. Confirm an allocation policy
   before launching any capacity measurement.
