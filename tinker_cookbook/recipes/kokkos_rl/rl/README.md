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
in `contree_images.json`. Set `resume_dir` to an interrupted timestamped result directory to
skip successful rollouts and retry only infrastructure errors.

Both Modal and ConTree rollout sandboxes disable network access by default. Image preparation
still has network access so repositories and build dependencies can be fetched. Only set
`allow_network=True` for tasks whose runtime contract explicitly requires external services;
doing so makes PR-derived benchmark scores vulnerable to solution lookup.

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

`sandbox_backend` selects `modal` (default) or `contree`, matching the evaluator. ConTree
caches prepared image UUIDs by Dockerfile digest, so `contree_cache_path` defaults to a
shared path rather than the run's log directory; point it at the cache an evaluation already
wrote to reuse those images instead of rebuilding them.

For both backends, agent rollout sandboxes have no network access by default. The exported
task image also flattens the repository to a single root commit and removes remotes, refs,
reflogs, and unreachable objects, preventing agents from mining the merged fix from Git
history. Re-export and republish the Harbor dataset whenever this clean-room setup changes.

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
