# OpenHands Apptainer workflow on Rivanna

## Components and ownership

Slurm allocates compute resources. The Cookbook driver runs within that allocation
and creates an `ApptainerSandbox` for each episode. OpenHands launches one agent
server inside each prebuilt SIF and executes the episode's commands through it.
The adapter does not allocate Slurm jobs or schedule containers across nodes.

One allocation can host multiple simultaneous episodes. Each receives a separate
container filesystem and server endpoint while sharing the allocation's CPU and
memory resources. Starting a new episode from a cached SIF does not require a new
Slurm allocation. A new allocation is needed when the existing resources or job
lifetime are insufficient.

Running the driver within the allocation keeps lifecycle control local. Running
it on another machine would additionally require remote creation/destruction,
endpoint discovery, authentication, connectivity, and capacity management; this
adapter does not implement that service.

## Prepare public task images

The adapter requires Apptainer and `openhands-workspace==1.45.0`. Image construction
is separate from execution: build each task's published environment with the
OpenHands agent-server runtime and a compatible entrypoint before invoking the
factory. Validate the complete server import and startup in the resulting image.

Each Harbor task environment must provide either `agent-server.sif` or a `sif.path`
file identifying its SIF. Relative pointer paths resolve against the environment
directory. Keep build definitions, dependency pins, image digests, and dataset
revisions so images can be recreated from the same public inputs.

The factory reads `environment.workdir` and `environment.gpus` from `task.toml`.
Explicit command working directories override the task default. GPU tasks require
an appropriate Slurm allocation; the adapter enables GPU passthrough and preserves
`CUDA_VISIBLE_DEVICES`, including MIG identifiers. This grants visibility to
allocated devices, not a per-sandbox GPU reservation.

Use a platform compatible with the published task configuration. Do not silently
alter source, tests, toolkit, or architecture settings to claim equivalent results.

## Runtime configuration

Inject `apptainer_sandbox_factory` into the Harbor evaluation driver. Allocate
Slurm resources before running the driver, and set its maximum concurrency to fit
those resources. Use the cluster's current documentation for partition, account,
wall-time, and device request syntax.

Image caches and SIFs may use scratch storage, which should be treated as
rebuildable rather than backed-up storage. Node-local temporary storage can avoid
shared-filesystem image extraction overhead. Keep durable build definitions and
results separately according to the site's retention policy.

CPU and memory are shared dynamically within the Slurm allocation. Task CPU
metadata does not enforce quotas. `command_env` can explicitly set cooperative
runtime controls such as `OMP_NUM_THREADS`; hard limits require additional
resource controls. Document the selected policy for any capacity comparison.

## Isolation and lifecycle

Startup reserves a process-local port lease and verifies a unique read-only
identity file through the responding server. This rejects an unrelated healthy
endpoint. Cross-process port coordination is not implemented.

The adapter disables default host-filesystem, configured bind-path, working-directory,
and home mounts. Only selected server environment variables are forwarded;
Tinker credentials must remain in the driver environment.

With `allow_network=False`, each command executes in a new user/network namespace.
Startup fails if the kernel cannot create that namespace. This does not isolate
the agent server's own host network and does not support network services shared
between successive commands. It is not a complete untrusted-code security boundary.

Cancellation during startup waits for the constructor thread and then retries
cleanup until it succeeds, retaining ownership through repeated cancellation.
New operations are disabled once cleanup begins; cleanup completion is recorded
separately so a transient failure can be retried. File uploads use bounded chunks
and a shared time budget; callers must check the result because failed uploads
can leave partial files.

## Evaluation timeouts and evidence

Harbor's `grader_timeout` controls the verifier command. The additional agentic
preset grading timer is disabled for this integration so it cannot interrupt a
longer configured verifier budget while also counting candidate capture and upload.
Container lifetime and individual command timeouts still apply. This introduces
no additional Tinker sampling timeout or retry wrapper.

The first attempt retains the usual rollout directory. Retries reserve independent
`attempts/NNN` directories. Once a retry completes, original root evidence is
archived under `attempts/000` and root-level relative links expose the completed
attempt to existing donor and archive consumers. The trajectory link is published
last. Failed attempts remain available for diagnosis.

When only grading fails and a complete candidate was saved, regrade that same
candidate in a fresh matching image. Verify its base revision and patch digest,
retain the original termination policy, and record recovery separately. Resampling
would change the candidate associated with the original evaluation slot.

For capacity methodology, see [Measuring Kokkos sandbox capacity](RIVANNA_KOKKOS_CAPACITY.md).
