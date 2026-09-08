# Measuring Kokkos sandbox capacity on Slurm

This guide describes a reproducible capacity experiment using task assets available
to the operator. It does not establish a universal sandbox concurrency limit.

## Prepare the workload

1. Choose a versioned public Harbor task dataset and record its revision. Use
   the task's published environment, source revision, tests, and baseline patches.
2. Build one agent-server SIF per task following [the workflow](RIVANNA_DESIGN.md).
   Record the image digest, build definition, dependency versions, and task mapping.
3. Validate the server import and startup, task working directory, a no-change
   baseline, and the task's documented reference solution before measuring capacity.
   An infrastructure error is not a valid failed-test result.
4. Use GPU tasks only when the allocation, driver, toolkit, and published task
   configuration are compatible. If a task requires a different platform, select
   a compatible allocation. Any task adaptation should be published separately
   with its exact configuration and reported as a different workload.

## Measure concurrency

Run fresh containers at increasing concurrency within one allocation. Start all
containers before releasing the grading wave. Keep the task mix, thread settings,
verifier timeout, and storage placement fixed across waves.

Record:

- Allocated CPUs, memory, GPU resources, node class, and process/file limits.
- Startup latency, successful grading throughput, and cleanup completion.
- Cgroup memory usage, process/thread counts, and driver file descriptors.
- Errors by stage: image preparation, startup, execution, grading, and cleanup.

A requested CPU count in task metadata does not impose a per-container quota.
All containers share the enclosing allocation unless the operator adds explicit
resource controls. OMP/BLAS environment variables control cooperative runtime
threading; they do not reserve cores. Record any such settings with the results.

The Python executor also limits concurrent blocking workspace calls independently
of resident container count. Record its worker count and keep it fixed.

## Interpret results

Repeat successful boundary levels and investigate the first failing interval.
A highest successful concurrency is a lower bound for the tested workload, not a
hardware maximum or a guarantee for longer model-driven trajectories. Separate
cold image conversion from cached startup and grading time.

Reject waves with endpoint-identity failures, unsuccessful cleanup, or invalid
baselines. Do not interpret a port collision as a CPU or memory capacity limit.
Keep raw evidence outside the public source tree unless it is explicitly prepared
for publication and reproducible from the referenced public inputs.
