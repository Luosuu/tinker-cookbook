# Kokkos evaluation and RL

This package consumes an already exported Kokkos Harbor task directory. It reuses the
shared `harbor_rl` environment and rollout loop, so Kokkos-specific code only selects tasks
and defines convenient defaults.

## Baseline evaluation

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos \
  tasks_dir=data/kokkos/SWE-kokkos-bench-v2 \
  model_name=thinkingmachines/Inkling-Small:peft:262144 \
  thinking_effort=0.9 \
  sandbox_backend=contree \
  num_samples=8 \
  pass_at_k=1,4,8
```

The evaluator runs tasks concurrently and writes machine-readable trajectory and reward
artifacts to its configured log directory. Report pass rates together with the exact dataset
version, model identifier, agent scaffold, turn limit, tool-call limit, token budget,
temperature, and thinking effort.

ConTree evaluations prepare each task's exported Dockerfile once and branch isolated
sessions from that immutable image for subsequent samples. Prepared image UUIDs are cached
at `/tmp/tinker-examples/kokkos_rl/contree_images.json` by default, shared with training. Set
`resume_dir` to an interrupted timestamped result directory to skip successful rollouts and
retry only infrastructure errors.

Resuming validates the model, checkpoint, generation and sandbox settings, and hashes of
the selected tasks' instructions, configuration, environment, and tests. Task subsets and
sample counts may change; summaries include only the requested tasks and sample indices.
Legacy result directories without `eval_identity.json` require a new output directory,
because their task contents cannot be verified. Operational changes such as concurrency do
not invalidate a run; changing its model, task payloads, or rollout budgets does.

ConTree is the default backend for Kokkos evaluation, training, and task auto-annotation because
its prepared images can be reused at lower cost. Keep a run on one backend for interpretable
results. If retries confirm a ConTree infrastructure failure, rerun only the affected task IDs
with `sandbox_backend=modal` and report the backend split explicitly.

Both Modal and ConTree rollout sandboxes disable network access by default. Image preparation
still has network access so repositories and build dependencies can be fetched. Only set
`allow_network=True` for tasks whose runtime contract explicitly requires external services;
doing so makes PR-derived benchmark scores vulnerable to solution lookup.

### OpenAI Responses evaluation

`eval_kokkos_openai` provides the same task and grader interface for OpenAI Responses API
models. Stateful tool-use responses bill the input context again on every turn, so the final
response's token count is not the run's total input usage. The evaluator records cumulative
input, cached input, cache-write input, output, reasoning tokens, and estimated cost.

The defaults are deliberately bounded: medium reasoning, 24 turns, 48 tool calls, 32K sampled
tokens, 12,000 characters returned to the model per tool result, and a $0.75 estimated per-task
cost cap. Adjust the four per-million-token price fields when provider pricing changes. Treat
the cost cap as an estimate and provider billing as authoritative; a request already in flight
can cross the cap before the evaluator stops the next turn.

The per-task cost budget is shared across failed attempts, retries, and resumed invocations.
Usage and cost totals include all attempts for the selected tasks, including failed grading;
pass rates use the final task outcomes. `num_attempts` records the total number of attempts.

## Reinforcement learning

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.rl.train_kokkos \
  tasks_dir=data/kokkos/SWE-kokkos-bench-v2 \
  model_name=thinkingmachines/Inkling-Small:peft:262144 \
  thinking_effort=0.9 \
  sandbox_backend=contree \
  eval_size=20 \
  eval_group_size=4 \
  remove_constant_reward_groups=True \
  group_size=4 \
  groups_per_batch=2
```

The training entrypoint builds grouped Harbor environments and delegates optimization to the
shared RL trainer. Advantages are centered within each task group. Use multiple rollouts per
task when estimating task difficulty: a single pass/fail observation is not a calibrated
success probability.

`sandbox_backend` selects `contree` (default) or `modal`, matching the evaluator. ConTree
caches prepared image UUIDs by Dockerfile digest, so `contree_cache_path` defaults to a
shared path rather than the run's log directory; point it at the cache an evaluation already
wrote to reuse those images instead of rebuilding them.

For both backends, agent rollout sandboxes have no network access by default. The exported
task image retains the original base commit as a shallow boundary and removes remotes,
refs, reflogs, and unreachable history. The verifier pins that commit's SHA and ignores Git
replacement refs, so committing changes cannot bypass protected-file checks.

The Kokkos training and evaluation entrypoints prepare temporary task copies with the current
environment and verifier generated from task metadata. Published payloads and historical
Oracle/NOP evidence remain unchanged; the effective task hashes are recorded for evaluation
resume checks. The new Dockerfile invalidates old sandbox image cache entries. These runtime
copies require fresh cloud validation before publishing a new dataset release; historical
release validation does not certify them. Direct Harbor users should re-export their tasks
with the current exporter to obtain the same fixed verifier.

### Held-out evaluation

`eval_size` holds out that many tasks, chosen by a `split_seed`-seeded shuffle, and trains on
the rest; `task_names` restricts the pool before that split. Held-out tasks are rolled out
`eval_group_size` times each and reported under `heldout/` in `metrics.jsonl`, separate from
the legacy in-sample `test/` metrics that `eval_size=0` still produces.

Sandbox concurrency is the binding constraint. Neither the training loop nor
`RLTestSetEvaluator` bounds it — a dataset's `batch_size` does not limit it, because the
evaluator flattens the whole dataset before rolling out — so effective concurrency is
`group_size × groups_per_batch` when training and `heldout_tasks_per_chunk × eval_group_size`
when evaluating. Cloud sandbox backends degrade sharply under load: on ConTree, grading
failures (`Image state ... cannot have state FAILED`) held at roughly 5% at concurrency 4 and
8 but rose to 33% at concurrency 16. Size both products to whatever your backend sustains.

Two related knobs matter for reward hygiene. `remove_constant_reward_groups=True` discards
groups whose trajectories all earned the same reward, which contribute no centered advantage —
useful when a large share of tasks are out of reach for the current policy.
`raise_on_grading_error=True` makes verifier infrastructure failures raise instead of scoring
`0.0`, so the rollout strategy (`MinViableGroup` by default for Inkling) retries or drops the
group rather than averaging a crashed grader into the baseline as a model failure.

Note that a task group counts as correct when *any* of its rollouts passes, so with
`eval_group_size > 1` the reported score is a pass@k, not pass@1.

Also mind the dataset contract: `HarborDataset` does not shuffle or wrap around, and training
runs a single pass, so the number of training tasks caps the number of steps
(`ceil(n_train / groups_per_batch)`).

Inkling models require an explicit `thinking_effort`; use the recommended renderer and a
long-context model variant for terminal trajectories. Do not impose client-side timeouts or
retry loops around Tinker sampling requests.

## Hosted Tinker Chat Completions evaluation

Use the OpenAI-compatible endpoint for independent, message-based evaluations. It accepts
both a base model identifier and a `tinker://.../sampler_weights/...` checkpoint as
`model_name`; a separate deployment is unnecessary.

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_tinker_chat \
  model_name=thinkingmachines/Inkling-Small:peft:262144 \
  task_names=kokkos__kokkos-6375 max_concurrency=1
```

The preset reads `TINKER_API_KEY` from the environment or `.env` and uses
`https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`.
It defaults to effort 0.9, temperature 1, 40 turns, 80 tool calls, 16,384 output tokens per
request and 65,536 per attempt. Input usage is cumulative across requests, capped at
5,000,000 before starting another request; the final request can cross this input threshold.
No whole-task retries run by default. The OpenAI SDK handles transient transport retries.
ConTree, protected Harbor grading and tool-output truncation use the existing evaluator.

Each task retains the complete returned assistant messages (including reasoning), tool
outputs, stop reasons and usage. Missing reasoning-token counts and unconfigured dollar
costs are `null`, not zero. Token and turn limits still apply. To enable a dollar budget,
explicitly set `estimate_cost=True`, all four verified price fields and
`max_cost_usd_per_task`; cost is an estimate and is checked between requests.
Use `resume_dir` only with the same model, endpoint, effort, budgets and task contents.

This endpoint is documented as beta for testing/evaluation; compatibility does not imply
higher throughput. See the [official API guide](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/).
Server rendering can differ from native renderers, so run a paired transport comparison
before replacing a research baseline. This path produces evaluation transcripts, not exact
sampled token IDs or training masks. Keep native Tinker collection for the self-training/RL
pipeline. The existing `eval_kokkos_openai` Responses API defaults remain available.
