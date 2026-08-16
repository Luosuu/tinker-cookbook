# Kokkos evaluation and RL

This package consumes an already exported Kokkos Harbor task directory. It reuses the
shared `harbor_rl` environment and rollout loop, so Kokkos-specific code only selects tasks
and defines convenient defaults.

## Baseline evaluation

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos \
  tasks_dir=data/kokkos/SWE-kokkos-bench-v2 \
  model_name=openai/gpt-oss-120b:peft:131072
```

The evaluator runs tasks concurrently and writes machine-readable trajectory and reward
artifacts to its configured log directory. Report pass rates together with the exact dataset
version, model identifier, agent scaffold, turn limit, tool-call limit, token budget,
temperature, and thinking effort.

## Reinforcement learning

```bash
uv run python -m tinker_cookbook.recipes.kokkos_rl.rl.train_kokkos \
  tasks_dir=data/kokkos/SWE-kokkos-bench-v2 \
  model_name=openai/gpt-oss-120b:peft:131072 \
  group_size=4 \
  groups_per_batch=8
```

The training entrypoint builds grouped Harbor environments and delegates optimization to the
shared RL trainer. Advantages are centered within each task group. Use multiple rollouts per
task when estimating task difficulty: a single pass/fail observation is not a calibrated
success probability.

Inkling models require an explicit `thinking_effort`; use the recommended renderer and a
long-context model variant for terminal trajectories. Do not impose client-side timeouts or
retry loops around Tinker sampling requests.
