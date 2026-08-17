# Replicating DeepCoder with Tinker

Competitive programming problems are a common testbed for RL with LLMs. The recent [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) blog post introduces a dataset and training pipeline for this purpose. This recipe demonstrates a similar setup using `Qwen3.5-4B` with thinking enabled.

## Running This Demo

### Sandboxing

Sandboxing is essential for safely executing generated code during training and evaluation. Three sandbox backends are supported:

#### SandboxFusion (Default)

[Sandbox Fusion](https://bytedance.github.io/SandboxFusion/) provides local Docker-based sandboxing. You can start a local sandbox in Docker with:

```bash
docker run -it -p 8080:8080 \
    -v ${TINKER_COOKBOOK_ROOT}/tinker_cookbook/recipes/code_rl/sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml \
    volcengine/sandbox-fusion:server-20250609
```

Here, `${TINKER_COOKBOOK_ROOT}` is the absolute path to your local `tinker-cookbook` repository. The training script reads the sandbox endpoint from the `SANDBOX_URL` environment variable. By default it uses `http://localhost:8080/run_code`. Example:

```bash
export SANDBOX_URL=http://localhost:8080/run_code
```

If you prefer not to use Docker, you can set up the sandbox manually by following the instructions in the [Sandbox Fusion repository](https://github.com/bytedance/SandboxFusion?tab=readme-ov-file#installation).

#### Modal (Alternative)

[Modal](https://modal.com/docs/guide/sandbox) provides cloud-based sandboxed execution without local Docker setup. To use Modal:

1. Install the modal extra and authenticate:
```bash
uv pip install 'tinker-cookbook[modal] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
modal token new
```

2. Set the sandbox backend in your training command:
```bash
python -m tinker_cookbook.recipes.code_rl.train \
    sandbox_backend=modal \
    ...
```

Optional environment variables for Modal:

- `MODAL_POOL_SIZE`: Number of concurrent sandboxes (default: 32)
- `MODAL_CREATION_RATE_LIMIT`: Max sandboxes created per second (default: 4)

#### Nebius ConTree

[ConTree](https://pypi.org/project/contree-sdk/) provides versioned cloud sandboxes. Set
either `NEBIUS_SANDBOX_API_KEY` (used by this repository) or the SDK-standard
`NEBIUS_API_KEY`, and set `NEBIUS_PROJECT_ID` to the project authorized for that key.
Then select the backend:

```bash
python -m tinker_cookbook.recipes.code_rl.train \
    sandbox_backend=contree \
    ...
```

`CONTREE_POOL_SIZE` controls the maximum concurrent executions (default: 32).

### Example command

Train a `Qwen3.5-4B` model with thinking enabled:

```bash
python -m tinker_cookbook.recipes.code_rl.train \
    model_name="Qwen/Qwen3.5-4B" \
    group_size=8 groups_per_batch=128 \
    learning_rate=4e-5 \
    lora_rank=32 \
    max_tokens=24576
```

## LiveCodeBench-CPP

The same recipe can train on NVIDIA's
[LiveCodeBench-CPP](https://huggingface.co/datasets/nvidia/LiveCodeBench-CPP). The dataset has
454 C++17 problems in the `v6_2408_2505` split and is licensed CC BY 4.0. It is large (about
1.9 GB) because it includes compressed private tests. The loader pins the dataset revision,
decodes those tests only when their RL batch is requested, and makes a deterministic, disjoint
train/evaluation split.

The C++ grader supports both dataset formats:

- AtCoder submissions are compiled once and run against stdin/stdout cases.
- LeetCode submissions are combined with the dataset-provided C++ test harness, then compiled
  and run.

All generated code still runs inside the selected sandbox. SandboxFusion uses its C++ runtime;
the Modal and ConTree adapters select images with `g++` installed.

Run a held-out baseline before training:

```bash
python -m tinker_cookbook.recipes.code_rl.evaluate \
    model_name="openai/gpt-oss-20b" \
    sandbox_backend=contree \
    eval_size=32 max_eval_examples=32 \
    max_tokens=32768 \
    output_path=/tmp/tinker-examples/livecodebench_cpp_gpt_oss_20b_baseline
```

Then train on the complementary problems while evaluating on the same held-out set at step 0
and every `eval_every` steps:

```bash
python -m tinker_cookbook.recipes.code_rl.train \
    dataset=livecodebench_cpp \
    model_name="openai/gpt-oss-20b" \
    sandbox_backend=contree \
    eval_size=32 \
    group_size=8 groups_per_batch=16 \
    learning_rate=4e-5 lora_rank=32 \
    max_tokens=32768 eval_every=10
```

Training on LiveCodeBench-CPP contaminates the trained checkpoint for reporting results on the
public LiveCodeBench-CPP benchmark. Use the held-out subset only for iteration diagnostics, and
use a separate, untouched benchmark for final model comparison.

After 190 steps of training, you can expect the following performance on **LiveCodeBench v6 (2025.02–2025.05)**:

| Model | Pass@1 | Pass@8 |
|-------|--------|--------|
| Qwen3.5-4B (before training) | 23.6% | 34.3% |
| Qwen3.5-4B (after 190 steps) | 52.4% | 76.0% |


[1] Luo, M., Tan, S., Huang, R., Patel, A., Ariyak, A., Wu, Q., Shi, X., Xin, R., Cai, C., Weber, M., Zhang, C., Li, L. E., Popa, R. A., & Stoica, I. (2025). DeepCoder: A fully open-source 14B coder at O3-mini level.
