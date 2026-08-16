# SWE-kokkos-bench

Published on Harbor Hub as
[`luosuu/SWE-kokkos-bench`](https://hub.harborframework.com/datasets/luosuu/SWE-kokkos-bench),
tagged `v1.0` and `latest`:

```bash
uvx harbor run -d luosuu/SWE-kokkos-bench@v1.0 -a <agent> -m <model>
```

SWE-kokkos-bench is a rolling collection of PR-derived coding tasks from the
Kokkos ecosystem. Each row in instances.jsonl carries the standard SWE-bench
fields (repo, instance_id, base_commit, problem_statement, patch, test_patch,
FAIL_TO_PASS, PASS_TO_PASS, hints_text, created_at, and version) plus executable
build metadata.

The current snapshot contains 30 tasks: 20 from `kokkos/kokkos`, eight from
`kokkos/kokkos-kernels`, and two from `kokkos/pykokkos`. Candidates from Kokkos
Tools, Remote Spaces, and Resilience remain in the mining corpus but are admitted
only after the same end-to-end validation; candidate count alone is not a release
criterion.

## Validation gate

Every accepted instance must demonstrate all of the following in a fresh Modal
sandbox:

1. the parent checkout configures, builds, and passes selected regression checks;
2. injecting only the hidden test patch produces the expected build/test failure;
3. applying only the production patch restores the build and all F2P/P2P checks;
4. the exported Harbor Oracle receives reward 1;
5. the empty-patch Harbor baseline receives reward 0.

Compiler-specific tasks use CUDA, HIP, or SYCL toolchain images without a GPU
unless execution really requires one. The verifier is offline, and agent runtime
network access is restricted to the configured model API.

validation-manifest.json records the validated JSONL source chosen for every
deduplicated instance. dataset.toml is the Harbor registry manifest.

## Baseline pass@1

On this 30-task snapshot, OpenAI `gpt-5.6-terra` scored 27/30 (90.0%), full
`thinkingmachines/Inkling:peft:262144` scored 17/30 (56.7%), and
`openai/gpt-oss-120b:peft:131072` scored 1/30 (3.3%). All runs had zero
exceptions. Inkling Small was not used for annotation or final evaluation.

## Local execution

Run Harbor with the Modal extra, point it at this directory, and select an agent
and model. The oracle and nop agents are useful for checking the reward bounds.
