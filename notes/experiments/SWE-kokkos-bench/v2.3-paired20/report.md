# SWE-kokkos-bench v2.3 paired 20-task evaluation

Date: 2026-09-01

Experiment commit: `8a91f34`

Dataset: local `SWE-kokkos-bench` v2.3 payloads

Task set: the exact 20 task IDs from the GPT-5.6-Terra comparison

## Result

| Model | Pre-v2.3 paired pass@1 | v2.3 pass@1 | Delta | Paired flips | v2.3 errors |
|---|---:|---:|---:|---:|---:|
| `zai-org/GLM-5.3:peft:262144` | 3/20 (15%) | 12/20 (60%) | +9 / +45 pp | 9 positive, 0 negative | 0 |
| `thinkingmachines/Inkling-Small:peft:262144` | 7/20 (35%) | 13/20 (65%) | +6 / +30 pp | 7 positive, 1 negative | 0 |

Inkling-Small also has eight historical samples per task. Across the same 20 tasks those old
payloads produced 53/160 successful rollouts, or 6.625 expected passes per 20-task draw. The new
single v2.3 draw produced 13 passes. Three new successes (`7088`, `7074`, and `7172`) had failed
all eight historical attempts, which is stronger evidence of a task-instruction effect than the
aggregate single-sample delta alone.

GLM's paired direction is especially consistent: all nine changed outcomes moved from failure to
success. An exact two-sided paired sign/McNemar test on the nine discordant tasks gives p=0.0039.
Inkling-Small's seven positive and one negative discordant outcomes give p=0.0703; its direction is
encouraging but one 20-task draw is not by itself a precise effect estimate.

## Configuration

Both v2.3 runs used temperature 1.0, 40 turns, 80 tool calls, 16,384 tokens per turn, 65,536
sampled tokens per trajectory, 114,688 total trajectory tokens, network-disabled Modal sandboxes,
concurrency 4, and a 900-second grader timeout. Infrastructure errors were retryable but incorrect
solutions were not. Both final summaries contain 20 valid results and zero errors.

- GLM renderer: automatically resolved `glm5_3_max_reasoning`.
- Inkling renderer: automatically resolved `tml_v0`; explicit thinking effort 0.9.
- GLM results: `glm-5.3/zai-org-GLM-5.3:peft:262144/20260901_080246/`.
- Inkling results: `inkling-small/thinkingmachines-Inkling-Small:peft:262144/20260901_080246/`.

The GLM run was much slower: median per-task elapsed time was 888 seconds versus 150 seconds for
Inkling-Small. These per-task times overlap under concurrency and should not be summed to estimate
wall-clock runtime.

## Paired outcomes

| Task | GLM old→v2.3 | Inkling old sample 1→v2.3 | Inkling old successes/8 |
|---|---:|---:|---:|
| `kokkos__kokkos-6375` | 0→1 | 1→1 | 7/8 |
| `kokkos__kokkos-6289` | 0→1 | 0→0 | 0/8 |
| `kokkos__kokkos-7030` | 0→1 | 1→1 | 7/8 |
| `kokkos__kokkos-6943` | 1→1 | 1→1 | 6/8 |
| `kokkos__kokkos-7040` | 0→1 | 1→1 | 4/8 |
| `kokkos__kokkos-7043` | 0→0 | 0→0 | 0/8 |
| `kokkos__kokkos-7088` | 0→0 | 0→1 | 0/8 |
| `kokkos__kokkos-7308` | 0→0 | 0→0 | 0/8 |
| `kokkos__kokkos-7293` | 1→1 | 0→1 | 2/8 |
| `kokkos__kokkos-7089` | 0→0 | 0→0 | 0/8 |
| `kokkos__kokkos-7151` | 0→1 | 0→1 | 1/8 |
| `kokkos__kokkos-7074` | 0→0 | 0→1 | 0/8 |
| `kokkos__kokkos-7172` | 0→1 | 0→1 | 0/8 |
| `kokkos__kokkos-7250` | 0→0 | 1→0 | 5/8 |
| `kokkos__kokkos-7248` | 0→1 | 0→1 | 6/8 |
| `kokkos__kokkos-7242` | 0→0 | 0→0 | 2/8 |
| `kokkos__kokkos-7148` | 0→1 | 0→1 | 1/8 |
| `kokkos__kokkos-7374` | 0→1 | 1→1 | 2/8 |
| `kokkos__kokkos-7327` | 1→1 | 1→1 | 8/8 |
| `kokkos__kokkos-7441` | 0→0 | 0→0 | 2/8 |

## Interpretation limits

The GLM comparison is the cleaner before/after: both evaluations used the same model, renderer,
temperature, rollout budget, Modal backend, and task IDs; the longer grader timeout only prevents
infrastructure failures and both compared records are valid graded trajectories.

The historical Inkling-Small run used ConTree and older pre-v2.3 task packaging, while the v2.3
run used Modal. Its eight-sample history helps quantify stochasticity, but the aggregate change
cannot be attributed exclusively to wording. A stronger follow-up would run at least four fresh
v2.3 samples on these 20 tasks with one backend and compare per-task success rates.
