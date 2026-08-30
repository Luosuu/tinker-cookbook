# Kokkos coding RL

This recipe contains two deliberately separate workflows:

- [`dataset/`](dataset/README.md) mines merged Kokkos pull requests, constructs
  verifier-backed tasks, validates them in clean sandboxes, and exports Harbor tasks.
- [`rl/`](rl/README.md) evaluates or trains Tinker models on an already exported Harbor
  dataset. It does not depend on the mining and annotation pipeline at runtime.

The released benchmark is available from Harbor as
`luosuu/SWE-kokkos-bench@v2.1`. Dataset construction is expensive and requires GitHub,
model-provider, and Modal credentials; normal evaluation and RL usage only require the
published tasks and the credentials required by the selected model provider.

```bash
uvx harbor run \
  -d luosuu/SWE-kokkos-bench@v2.1 \
  -a <agent> \
  -m <model>
```

See the two workflow READMEs for reproducible commands and implementation details.
