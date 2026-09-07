# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently two available backends: SandboxFusion for local execution and Modal for cloud execution.

Harbor also supports the experimental OpenHands/Apptainer adapter below.

### OpenHands / Apptainer (Harbor)

`apptainer_sandbox.ApptainerSandbox` implements `SandboxInterface` for a
persistent OpenHands agent server inside Apptainer. Use Python 3.12+ and install
`openhands-workspace==1.45.0`, `openhands-sdk==1.45.0`, and
`openhands-agent-server==1.45.0` in the client environment. Apptainer must be on
the client machine's PATH; on HPC, launch the client inside allocated resources.

Prepare each task's image separately, including the OpenHands agent server and
its entrypoint. Put the resulting SIF at `environment/agent-server.sif`, or put
its absolute path (or a path relative to `environment/`) in `environment/sif.path`.
The adapter does not build Dockerfiles or substitute a generic task image.

```python
from tinker_cookbook.sandbox.apptainer_sandbox import apptainer_sandbox_factory
from tinker_cookbook.recipes.harbor_rl.train import cli_main

await cli_main(config, tasks, sandbox_factory=apptainer_sandbox_factory)
```

The factory starts one fresh container per call. It keeps that environment
across commands until cleanup, enables fakeroot and Docker compatibility, and
disables VSCode to avoid its shared default port. Commands have explicit working
directories; shell-local variables and `cd` do not constitute persistent state.
`APPTAINER_CACHEDIR` controls the cache location.

This is an initial integration adapter, not a cluster scheduler or sandbox pool.
The caller controls concurrency and resource limits. The lifetime argument is
checked on command submission and limits each command to the remaining time;
callers must still clean up idle environments and bound their outer job lifetime.
Output is truncated on the client after execution, not bounded at the server.
Apptainer uses host networking: distinct API ports do not isolate arbitrary
network services launched by tasks. Real task-image compatibility must be tested.

## Backends

### SandboxFusion (local Docker)

[Sandbox Fusion](https://bytedance.github.io/SandboxFusion/) is a Docker-based code execution sandbox. Start a local sandbox in Docker with:

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

For RL workloads, you may want higher concurrency. See [`recipes/code_rl/sandbox_config/local.yaml`](../recipes/code_rl/sandbox_config/local.yaml) for an example configuration that can be mounted with `-v`, and see [`recipes/code_rl/README.md`](../recipes/code_rl/README.md) for instructions on using it.

If you prefer not to use Docker, see the [Sandbox Fusion repository](https://github.com/bytedance/SandboxFusion?tab=readme-ov-file#installation) for manual setup.

Example usage:

```python
from tinker_cookbook.sandbox import SandboxFusionClient

client = SandboxFusionClient()
success, response = await client.run(
    code="print('hello')",
    files={"data.txt": "some content"},
    timeout=30,
)
await client.close()
```

Environment variables:

- `SANDBOX_URL`: Endpoint URL (default: `http://localhost:8080/run_code`)
- `SANDBOX_MAX_CONCURRENCY`: Max concurrent requests (default: 4)

### Modal (cloud)

[Modal Sandboxes](https://modal.com/products/sandboxes) provide cloud-based isolated execution environments. Requires authentication with: `modal token new`

Example usage:

```python
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox, ModalSandboxPool

# Single sandbox (conforms to SandboxInterface)
sandbox = await ModalSandbox.create()
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python /workspace/code.py", workdir="/workspace")
print(result.stdout)
await sandbox.cleanup()

# Pool for concurrent execution (recommended for RL workloads)
pool = ModalSandboxPool(pool_size=32)
result = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)
print(result.stdout)
```

Environment variables:

- `MODAL_POOL_SIZE`: Number of sandboxes in the pool (default: 32)

### Nebius ConTree (cloud)

ConTree implements persistent sandbox state as a chain of immutable image versions:

```python
from tinker_cookbook.sandbox.contree_sandbox import ContreeSandbox

sandbox = await ContreeSandbox.create(image="python:3.12-slim")
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python /workspace/code.py")
await sandbox.cleanup()
```

Authentication accepts `NEBIUS_SANDBOX_API_KEY`, `NEBIUS_API_KEY`, or the standard
ConTree profile variables. IAM keys also require the authorized `NEBIUS_PROJECT_ID`.
`CONTREE_POOL_SIZE` sets Code RL grading concurrency.
