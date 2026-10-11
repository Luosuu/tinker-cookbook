# HPC guidance as native Pi resources

The Pi evaluator keeps the benchmark task instruction unchanged. Choose an explicit
integration arm with `hpc_mode=baseline|prompt|skill|hook`; the default is `baseline`.
Do not call `apply_profile()` on tasks passed to Pi. That old instruction-prefix
approach is retained only for reproducing earlier native-loop experiments.

| Mode | Pi mechanism | When guidance enters context |
|---|---|---|
| baseline | No HPC resources | Never |
| prompt | `DefaultResourceLoader` prompt template and native `/hpc-debug` expansion | A separate user message at session start |
| skill | Native `SKILL.md` discovery | Name/description initially; body only if the model reads it |
| hook | Extension `before_agent_start` and `turn_end` events | Initial workflow plus reminders at 25%, 50%, 75% of the turn budget |

All three treatments use the same canonical workflow body from
`pi-hpc-agent/skills/hpc-debug/SKILL.md` in the separate agent checkout. The prompt template is
materialized from that body, registered with Pi's resource loader, and invoked by
`session.prompt('/hpc-debug')`. The original task is a separate unchanged user
message: no argument substitution can alter its quotes, newlines, or `$` symbols.

The skill arm advertises a real Pi skill and installs its file at
`/opt/hpc-harness/skills/hpc-debug/SKILL.md` inside the sandbox. Pi's native skill
instructions let the model load it with `bash`. There is no automatic full-body
injection or extra host filesystem tool. Not selecting the skill is an observable
outcome, not a reason to force-load it silently.

The hook arm uses explicit in-process Pi extension factories; it does not patch the
provider request payload to add HPC guidance. Its reminders are advisory, do not
claim tests succeeded, and do not force continuation after a natural answer. They
never increase the model/tool budgets. The hook arm deliberately has repeated
reminders, so it is a comparison of mechanisms/policies, not token-identical exposure.

## Runtime and isolation

Pi 0.87.0 runs on the orchestration host with model API access. The only enabled tool
is the same custom sandbox `bash` in all original arms (`plugin_profile=none`).
The separate `plugin_profile=todo` treatment enables the pinned official Pi todo tool;
use `hpc_mode=hook plugin_profile=todo` for the new combined configuration. All tool calls
share the total tool budget. Plugin identity and source hashes are journaled. Personal extensions, skills, context
files, and prompt templates are disabled; only explicitly selected packaged resources
are loaded. Provider configuration, accounting and limits remain in the runner.

No task solutions or hidden tests are used as guidance. Each episode receives a fresh
network-disabled sandbox, captures a candidate before hidden tests are uploaded, and
uses the existing Harbor verifier. Pi request/message events and actual hook events
are saved. `skill_path_commands` counts commands mentioning the skill path; inspect
their results to establish whether the body was successfully read. It is not a claim
that the model understood or followed the skill.

## Independent agent package

The runtime is maintained in a separate `pi-hpc-agent` checkout beside this repository.
This repository owns benchmark loading, sandbox execution, grading and trace collection;
the agent owns the Pi loop and native HPC resources. It has no Python/cookbook dependency.
Set `pi_agent_dir=/absolute/path/to/pi-hpc-agent` on the evaluator or set
`PI_HPC_AGENT_DIR`. Explicit configuration takes precedence over the environment;
otherwise the adapter looks for the sibling checkout. A missing package fails clearly.
Rivanna drivers pass the same `pi_agent_dir` into `CLIConfig`, with a versioned checkout
on the orchestration host. Historical frozen experiment snapshots remain self-contained.

## Install and run

Requires Node 22+ supported by the pinned SDK, cookbook Python dependencies, the
optional `openai` dependency used by shared chat configuration, a model API key, and
an authenticated sandbox backend.

```bash
cd ../pi-hpc-agent
npm ci --ignore-scripts --no-audit --no-fund
export PI_HPC_AGENT_DIR="$PWD"
cd ../tinker-cookbook
uv run --with openai python -m tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_pi \
  hpc_mode=skill task_names=YOUR_FIXED_DEVELOPMENT_TASK num_samples=1 \
  max_concurrency=1 output_path=notes/experiments/harness/pi-skill
```

The standalone CLI uses ConTree. Rivanna orchestration passes its frozen runtime task
and `apptainer_sandbox_factory` directly to `evaluate_task`; it must not regenerate a
Dockerfile and pretend an old SIF was rebuilt. `pi_rollout` installs the skill when
needed, so both backends use the same integration. Record the SIF hash, frozen task
identity, CPU/memory allocation and original task instruction hash in the run manifest.

Use independent output directories for each mode. For an initial comparison, run the
same development task once per arm, with matched weights, effort, temperature, resources,
output limits and budgets. Start every arm from the same immutable image, and do not
feed earlier candidates or grades to later arms. A single sample per arm checks behavior;
use more tasks and samples before selecting a winner.

Inspect actual skill reads, hook firings, first build/test command, natural completion,
verifier outcome, infrastructure errors and cumulative token usage. Instructions and
resource hashes are independently recorded, separating the task from the harness.
The historical instruction-prefix pilot is not one of these native arms.

## Training and reproducibility

Pi Chat rollouts remain evaluation/mining evidence only (`eligible_for_training=false`):
they do not contain original sampled Tinker token IDs and logprobs. A sampler checkpoint
URL is not a Chat Completions model ID. Do not claim a matched-checkpoint comparison
without confirming both transports serve the same weights.

The existing `harness_profile` option on native Tinker eval/RL/self-training entrypoints
is the older prompt-prefix experiment, not an implementation of these Pi mechanisms.
Training on Pi-native interactions will require a token-preserving provider adapter or
an explicitly designed offline data path; this change does not silently substitute
re-tokenized Chat traces into RL.

`harness.json` and evaluation identity include the selected mode and hashes of the
runner, integration module, canonical skill and package manifest. Keep original transcripts
and patches. The Rivanna pilot exposed OpenHands `workspace/bash_events` in raw candidate
patches; these artifacts require cleanup/isolation work before any training admission.

## Tests

```bash
cd ../pi-hpc-agent
npm test
```

Tests exercise the actual Pi SDK against a local mock SSE endpoint: all four modes,
unchanged task messages, identical tool schemas, lazy skill exposure, hook delivery,
and budget termination. Python tests cover sandbox skill installation, rejection of
pre-prefixed tasks, resource identity, candidate-before-verifier ordering, and failures.
These checks do not establish benchmark performance.

API references:
[skills](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/skills.md),
[prompt templates](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/prompt-templates.md),
[extensions](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md).
