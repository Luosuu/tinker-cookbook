---
pretty_name: SWE-kokkos-bench
language:
- en
license: other
task_categories:
- text-generation
tags:
- code
- software-engineering
- reinforcement-learning
- terminal-agent
- swe-bench
- kokkos
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: instances.jsonl
---

# SWE-kokkos-bench

SWE-kokkos-bench is a public, verifier-backed benchmark of 100 repository-level
software-engineering tasks mined from merged pull requests in the Kokkos ecosystem. Each task
starts from the parent revision of a real pull request and asks an agent to implement the
corresponding change. Correctness is checked by task-specific build and regression commands
against a held-out test patch.

The dataset supports two complementary interfaces:

- this Hugging Face repository exposes one SWE-bench-compatible JSONL row per task for
  analysis, filtering, and integration with dataset tooling;
- the executable Harbor release packages repository environments, protected graders, and
  task metadata for agent evaluation and reinforcement learning.

Version 2.2 contains the unchanged 100 task environments and verifiers from v2.1, with
clean-room instructions that explicitly direct agents away from unavailable network and Git
history lookup paths and toward local source diagnosis and focused public verification.

## Links

- [Executable dataset on Harbor](https://hub.harborframework.com/datasets/luosuu/SWE-kokkos-bench/latest)
- [Analysis dataset on Hugging Face](https://huggingface.co/datasets/luosuu/SWE-kokkos-bench)
- [Public `tinker-cookbook` fork](https://github.com/Luosuu/tinker-cookbook/tree/science-rl)
- [Dataset construction code](https://github.com/Luosuu/tinker-cookbook/tree/science-rl/tinker_cookbook/recipes/kokkos_rl/dataset)
- [Evaluation and RL code](https://github.com/Luosuu/tinker-cookbook/tree/science-rl/tinker_cookbook/recipes/kokkos_rl/rl)

## Quick start

Load the metadata with `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("luosuu/SWE-kokkos-bench", split="test")
print(dataset.num_rows)  # 100
print(dataset[0]["instance_id"])
print(dataset[0]["problem_statement"])
```

Run the executable benchmark through Harbor:

```bash
uvx harbor run \
  -d luosuu/SWE-kokkos-bench@v2.2 \
  -a <agent> \
  -m <model>
```

Use the immutable `v2.2` tag for reported results. `latest` currently points to the same
revision but may move in the future.

## Dataset composition

| Source repository | Tasks |
| --- | ---: |
| `kokkos/kokkos` | 90 |
| `kokkos/kokkos-kernels` | 8 |
| `kokkos/pykokkos` | 2 |
| **Total** | **100** |

The verifier signal is a compile failure for 76 tasks and a runtime test failure for 24
tasks. All packaged tasks use CPU validation environments; tasks requiring unsupported GPU
execution were not admitted to this release.

The dataset has a single `test` split. It is intended for evaluation, RL environment
construction, and research on software-engineering agents—not as a conventional supervised
train/test benchmark with a hidden public leaderboard split.

## Task format

`instances.jsonl` contains one object per task. Important fields are:

| Field | Meaning |
| --- | --- |
| `repo` | Upstream GitHub repository, such as `kokkos/kokkos`. |
| `instance_id` | Stable identifier derived from repository and pull-request number. |
| `base_commit` | Parent revision checked out when the task begins. |
| `problem_statement` | User-visible issue or pull-request description, sanitized to avoid solution leakage. |
| `patch` | Full merged pull-request diff, retained for research compatibility. |
| `code_patch` | Production portion of the gold change. |
| `test_patch` | Held-out test portion injected by the verifier. |
| `FAIL_TO_PASS` / `fail_to_pass` | Checks expected to fail before and pass after the fix. |
| `PASS_TO_PASS` / `pass_to_pass` | Regression checks expected to remain passing. |
| `configure_command` | Command that configures the parent checkout. |
| `build_command`, `build_targets` | Narrow build command or targets used for validation. |
| `f2p_commands`, `p2p_commands` | Exact executable checks used by the verifier. |
| `changed_files` | Structured metadata for files changed by the source pull request. |
| `metadata` | Build system, toolchain, failure stage, accelerator requirements, and source URL. |

The uppercase and lowercase F2P/P2P fields are intentionally both present: uppercase fields
preserve SWE-bench compatibility, while lowercase fields retain the recipe's typed source
representation.

### Harbor task layout

The Harbor release contains one directory per task:

| Path | Purpose |
| --- | --- |
| `instruction.md` | Problem statement shown to the agent. |
| `task.toml` | Task identity, environment, verifier, and resource configuration. |
| `environment/Dockerfile` | Reproducible repository checkout and build environment. |
| `tests/test.sh` | Offline grader entrypoint. |
| `tests/test.patch` | Held-out test changes injected only during grading. |
| `solution/gold.patch` | Oracle production patch used for release validation. |
| `metadata.json` | SWE-compatible source row and construction metadata. |

The agent receives the instruction, repository checkout, shell tool, and pre-warmed build
tree. `tests/` and `solution/` are not copied into the agent image. Harbor submits only the
agent's repository patch to the clean grader, which returns binary reward 1 for a fully
passing solution and 0 otherwise. The dataset metric is mean reward across the selected
tasks, with missing task results treated as zero.

## How tasks were constructed

The construction pipeline is available in the public
[`tinker-cookbook`](https://github.com/Luosuu/tinker-cookbook/tree/science-rl/tinker_cookbook/recipes/kokkos_rl)
fork and is separated into `dataset/` and `rl/` packages.

1. Query merged pull requests from supported Kokkos ecosystem repositories.
2. Keep changes containing both production code and unit-test modifications, then partition
   each pull-request diff into a production patch and a held-out test patch.
3. Recover a problem statement from linked issues or the pull-request description and remove
   obvious implementation leakage.
4. Propose narrow build targets and fail-to-pass/pass-to-pass commands.
5. Execute the proposal in a fresh Modal sandbox and retain only tasks satisfying all
   required state transitions.
6. Export validated rows into Harbor task directories with protected tests and offline
   grading.

Mining rejects documentation-only changes, dependency bumps, changes without both production
and test modifications, oversized diffs, unsupported source layouts, and tasks whose required
runtime cannot be reproduced in the release environment. Candidate counts are not a release
criterion; only end-to-end validated tasks are included.

## Validation and grading

Every accepted task demonstrated these transitions in a fresh sandbox:

1. the parent checkout configures and its selected targets build;
2. selected regression checks pass on the parent;
3. applying only the held-out test patch produces the expected compile or runtime failure;
4. applying the production patch restores all fail-to-pass and pass-to-pass checks;
5. the exported Harbor Oracle agent receives reward 1;
6. the exported Harbor NOP agent receives reward 0.

For v2.0, all 100 Oracle trials returned reward 1 and all 100 NOP trials returned reward 0,
with no infrastructure exceptions. The 30 unchanged v1.0 task payloads retained identical
Harbor digests; the 70 additions were independently validated for the v2.0 release.

The agent cannot rely on modifying the scoring apparatus. Before grading, the verifier
rejects or restores protected test, CMake-registration, and CI paths, injects the held-out
test patch, and runs without network access.

## Baseline results

Pass@1 results on the 70 tasks added in v2.0:

| Model | Effort | Turns | Tool calls | Passed | Pass rate | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thinkingmachines/Inkling:peft:262144` | 0.99 | 40 | 60 | 36/70 | 51.4% | 0 |
| `thinkingmachines/Inkling-Small:peft:262144` | 0.9 | 40 | 80 | 42/70 | 60.0% | 0 |
| `openai/gpt-oss-120b:peft:131072` | not applicable | 20 | 40 | 4/70 | 5.7% | 0 |

All three runs used temperature 1.0, a maximum of 16,384 tokens per model turn, a
112K-token trajectory cap, and a 64K sampled-token cap. Tasks were evaluated once and in
parallel using the same terminal-agent scaffold and clean Harbor grader.

Full Inkling and GPT-OSS used settings directly comparable to their v1.0 evaluations. Their
combined scores over all 100 v2.0 tasks are respectively 53/100 (53.0%) and 5/100 (5.0%).
Inkling Small has no complete evaluation on the unchanged 30-task subset, so a 100-task score
is not reported for that model.

These are single-rollout observations, not calibrated per-task success probabilities. Exact
scores depend on the model revision, renderer, agent scaffold, temperature, thinking effort,
turn limit, tool-call limit, and trajectory token budget. Report all of those settings with
benchmark results.

## Intended uses

- evaluate terminal-based coding agents on C++ and HPC-library maintenance;
- construct grouped, verifier-reward environments for reinforcement learning;
- study task difficulty, model complementarity, tool use, and failure modes;
- reproduce or extend the PR-to-verifiable-task construction pipeline.

For RL, sample multiple trajectories per task. Tasks that always pass or always fail within a
group provide little centered-advantage signal; a one-shot baseline is only a coarse guide for
curriculum design.

## Limitations and risks

- **Public-source contamination:** every task comes from a public merged pull request. A model
  may have seen the patch or surrounding discussion during pretraining.
- **Gold data is public:** `patch`, `code_patch`, and `test_patch` support research and
  reproducibility. Evaluation harnesses must prevent the agent from reading dataset metadata
  or solution files during a rollout.
- **Domain concentration:** 90% of tasks come from Kokkos Core. Results should not be treated
  as a general estimate of software-engineering ability.
- **Failure-mode skew:** 76% of tasks use compile failures, so runtime semantics and
  performance regressions are underrepresented.
- **Platform coverage:** this release emphasizes reproducible CPU toolchains and does not
  measure the full CUDA, HIP, SYCL, or multi-node behavior of the Kokkos ecosystem.
- **Stochastic evaluation:** pass@1 can move with sampling and rollout budgets. Multiple seeds
  or pass@k are needed for robust task-level difficulty estimates.

## Licensing and attribution

The dataset construction code in `tinker-cookbook` is Apache-2.0. The task content is derived
from public Kokkos ecosystem repositories and pull requests; source files, patches, issue text,
and tests remain subject to their respective upstream licenses and attribution requirements.
Because the Hub repository contains material from multiple upstream projects, its metadata is
marked `license: other`. Consult the source URL in each row's `metadata.html_url` and the
corresponding upstream repository before redistribution or commercial use.

## Versioning

- `v1.0`: 30 tasks.
- `v2.0`: 100 tasks, consisting of the unchanged v1.0 payloads plus 70 new Kokkos Core tasks.
- `v2.1`: the same 100 task specifications and verifiers as v2.0, with clean-room runtime
  packaging that removes Git history and disables agent sandbox network access.
- `v2.2`: the same environments and verifiers as v2.1, with instructions that state those
  clean-room constraints and prioritize local diagnosis, production edits, and focused public
  build/test targets.

When citing results, use the dataset name, immutable version, model identifier, and complete
rollout configuration. A suggested textual citation is:

> SWE-kokkos-bench v2.2, a 100-task verifier-backed benchmark mined from merged Kokkos
> ecosystem pull requests, released through Harbor and Hugging Face in 2026.
