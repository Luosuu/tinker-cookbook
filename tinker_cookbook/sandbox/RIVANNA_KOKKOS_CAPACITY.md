# Rivanna Kokkos sandbox capacity experiment

Status: stopped after an infrastructure failure (2026-09-08). Complete Oracle
grading passed through concurrency 128. At 256, a server port collision and a
patch-application failure made the wave invalid; the allocation was canceled.
128 is the highest tested successful level, not an established maximum.

## Code and workload

The Apptainer feature branch was rebased onto the fork's `science-rl` commit
`d23367e`. The adapter now honors `environment.workdir` from `task.toml`, while
allowing explicit command directories to override it. The adapter, Harbor tool,
and Kokkos training configuration regression suites passed: 36 tests.

The selected tasks are `kokkos__kokkos-9309`, `kokkos__kokkos-9414`, and
`kokkos__kokkos-9344` from `data/kokkos/SWE-kokkos-bench-v2`. Task preparation
uses `prepared_kokkos_tasks`, including its current baseline-pinned verifier.
Each task retains its own source revision and prebuilt targets. Ubuntu 24.04
images include OpenHands Agent Server and SDK 1.45.0.

This is a sandbox capacity experiment using Oracle production patches and
HarborReward, not a model-quality evaluation or an RL optimizer run. Planned
concurrency levels are 3, 8, 16, 32, 64, 128, and 256, with synchronized grading
after all containers have started. NOP validation precedes Oracle validation.
An infrastructure failure must not count as a valid zero-reward NOP result.

No per-container CPU or memory limit, CPU binding, or runtime build-parallelism
override is applied. The current generated verifier defaults to one build job;
this is distinct from limiting the entire container to one CPU. Image preparation
uses eight build jobs; that setting is not retained in the runtime environment.

## Results so far

- The BII large-memory partition accepted a single-node allocation of 128 CPUs
  and 900 GiB. The driver observed 128 allowed CPUs and a 65,536 file-descriptor
  limit. This establishes allocation availability, not sandbox capacity.
- Building on the shared scratch filesystem stalled for minutes while unpacking
  Boost headers. The observed `dpkg` process was in I/O wait. Rebuilding with
  node-local `/tmp` as `APPTAINER_TMPDIR`, while retaining SIF/cache files on
  scratch, completed all three images in 5 minutes 4 seconds. These are different
  preparation attempts, not a controlled storage benchmark.
- The first self-built runscript incorrectly assumed the official Docker image's
  `openhands-agent-server` launcher was provided by the PyPI distribution. It was
  replaced by `python -m openhands.agent_server`.
- After that correction, startup failed while importing `libtmux`. The remaining
  image fix is to install the terminal runtime dependencies (`tmux`, `libtmux`)
  and validate `import openhands.agent_server.api` before packaging. On resumption,
  the image also needed `openhands-tools==1.45.0`. Pinning `browser-use==0.11.0`
  avoids the newer browser package's incompatible OpenAI SDK pin. All three
  repaired images now pass `pip check` and the complete server import check.
- Rivanna reports a default Apptainer session directory limit of 64 MiB. No writable-layer
  space failure was observed in the completed waves for these three tasks.
- SSH connectivity to all three DNS addresses for the login service subsequently
  timed out. The terminal dependency fix was prepared locally but could not be
  uploaded. All submitted build/precheck jobs had already ended.
- SSH access recovered on 2026-09-08. The repaired images were tested in a new
  128-CPU, 900-GiB allocation. All three NOP rewards were 0, and all three Oracle
  rewards were 1. Cached Oracle startup took approximately 13–14 seconds per
  container. Oracle grading took 47.4 seconds (9309), 75.1 seconds (9414), and
  567.1 seconds (9344). Cleanup passed for all three containers.
- The tested node has two AMD EPYC 7763 processors, 64 physical cores per socket,
  SMT disabled, and eight NUMA nodes. Its Slurm service has `pids.max=max`.

### Completed Oracle waves

| Concurrent sandboxes | Passed | Whole-wave time (seconds) | Median startup (seconds) | Sampled cgroup peak (GiB) |
| --- | --- | --- | --- | --- |
| 3 | 3/3 | 584.7 | 14.3 | 7.85 |
| 8 | 8/8 | 599.2 | 14.6 | 15.86 |
| 16 | 16/16 | 605.1 | 15.4 | 31.05 |
| 32 | 32/32 | 618.8 | 18.0 | 62.55 |
| 64 | 64/64 | 780.1 | 25.1 | 111.64 |
| 128 | 128/128 | 1361.7 | 38.9 | 221.97 |

Wave time includes startup, grading, and cleanup. Memory is sampled every two
seconds from the batch task's memory cgroup, including its descendants and cache;
it is not a per-container reservation. Each wave uses fresh containers, distributed
round-robin across the three tasks, and retains them until the wave is cleaned up.

At concurrency 128, all containers started in 64.9 seconds, every Oracle reward
was 1, and cleanup passed. The sampled peak was 5,977 descendant threads and 265
driver file descriptors. This level has not been repeated.

At concurrency 256, constructors and basic command probes completed in 134.4
seconds. The server log nevertheless recorded an address-in-use failure on port
34182. A subsequent Oracle patch failed to apply in replica 79. Together these
indicate possible endpoint cross-routing: the adapter shares a session key and
relies on upstream free-port selection and generic health checks, which do not
prove that a responding server belongs to the requested container. No 256-wave
reward or isolation result is considered valid. The run was canceled and no
stress allocation remained active when SSH was rechecked. Observed memory in
this incomplete wave does not establish a hardware capacity limit.

Before resuming, prevent port reuse among initializing/live workspaces and verify
a unique container identity before returning an adapter. Validate isolation under
concurrent startup, then repeat complete grading near the boundary. This fix is
pending; the results above describe the original startup implementation.

The default Python executor used by `asyncio.to_thread` also bounds simultaneous
blocking operations (32 worker threads on the tested host). Resident sandbox
count therefore differs from the number of simultaneously executing tool calls.
This behavior is retained for the default-policy scan.

## Resume and reporting requirements

1. Server import/startup, task working directory, NOP reward 0, and Oracle reward
   1 have passed for all three selected tasks. Retain those baseline artifacts.
2. Diagnose any later writable-layer failures separately from CPU/memory pressure.
3. Run increasing concurrency on the same node class. Record startup latency,
   successful grading throughput, cleanup, cgroup memory usage, process/thread
   counts, and driver file descriptors. RSS sums are diagnostic only because
   shared pages can be counted repeatedly.
4. Repeat successful boundary levels and refine the first failing interval in a
   clean allocation. A highest tested successful level is a lower bound, not a
   universal maximum for the 100-task dataset or for model-driven RL trajectories.
5. Keep network-isolation limitations explicit: the current OpenHands adapter
   uses host networking and does not enforce the benchmark's offline policy.
   Oracle capacity testing does not establish clean-room model evaluation parity.

## GPU/MIG validation (2026-09-08)

The snapshot marks two tasks as requiring GPU execution: `kokkos__kokkos-9147`
and `kokkos__kokkos-8989`. This experiment validated **9147 only**, with the
current baseline-pinned Harbor verifier and production Oracle patch. It did
not perform model sampling or an RL optimizer update.

The live `gpu-mig` partition exposed RTX PRO 6000 Blackwell `1g.24gb` slices.
One slice, four CPUs, and 32 GiB host RAM were requested. The original task
image used CUDA 12.8.1 and `Kokkos_ARCH_ADA89`; the private prepared copy used
CUDA 12.9.1 and `Kokkos_ARCH_BLACKWELL120` in both environment and verifier.
Source revision, instructions, production patch, and hidden tests were preserved.
Published dataset files were not rewritten. The adapter passed `enable_gpu=True`
and forwarded the Slurm-provided `CUDA_VISIBLE_DEVICES=MIG-...` unchanged.

| Fresh-container trial | Reward | Startup (s) | Grading (s) | Whole wave (s) |
| --- | --- | --- | --- | --- |
| NOP | 0 | 11.94 | 3.28 | 17.91 |
| Oracle | 1 | 6.04 | 13.57 | 22.28 |
| Oracle repeat | 1 | 6.04 | 13.46 | 22.17 |

NOP failed compilation because the base implementation lacks `cuda_node()`,
the intended task defect. Both Oracle trials compiled the patched test and
executed all six `cuda_GraphInterOp` tests, including `interact_with_cuda_node`,
with no skips or failures. CTest runtime was approximately 0.36 seconds in the
first Oracle trial. Cleanup passed after every trial. The validation allocation
completed successfully in 1 minute 7 seconds; image construction on a CPU node
took 5 minutes 44 seconds. No claim about queue-time improvements follows from
this single run.

Versions: Apptainer 1.4.5; OpenHands 1.45.0; CUDA compiler 12.9.86; host NVIDIA
driver 595.71.05. SIF SHA-256:
`58b2823f11e5f9f3dcc066d8d8449489a73b842ac8affa11695805aeb8a70720`.
Adapter, Harbor tool, and Kokkos training configuration regression tests: 45 passed.
The second GPU task and multiple sandboxes sharing a MIG slice remain untested.

### Training concurrency interpretation

For the three measured CPU tasks, 64 resident rollouts is a reasonable first
training trial after fixing the endpoint collision; 128 is a subsequent trial,
not a production guarantee. At `group_size=4`, 16 or 32 groups correspond to
64 or 128 rollout slots per batch. Account separately for any overlapping evals
and asynchronous batches. Default Python thread-pool capacity limits simultaneous
blocking workspace calls to 32; it does not cap Tinker sampling at 32.

The endpoint collision observed at 256 can occur probabilistically at lower
concurrency, so successful lower waves do not remove that correctness issue.
Resolve it before long-running training. Real trajectories must also establish
memory, latency, timeout, and reward behavior under representative tool use.
GPU tasks require a separate concurrency budget; this single-slice validation
supports neither 64 nor 128 simultaneous GPU rollouts. The current local factory
does not route CPU-job requests into separate GPU allocations.


## Follow-up before Inkling evaluation

Port leases outside the tested node's ephemeral client range and per-container
startup identity checks have been implemented. A fresh 64-container test passed
unique ports, independent same-path file contents, blocked external networking,
and cleanup. The optional offline mode isolates each command's network namespace;
it does not isolate the Agent Server's own network. Relevant regression suites
passed 48 tests. This does not retroactively validate the failed 256 wave or
establish a new capacity maximum.
