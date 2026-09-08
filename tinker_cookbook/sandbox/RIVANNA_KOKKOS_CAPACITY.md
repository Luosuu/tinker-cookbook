# Rivanna Kokkos sandbox capacity experiment

Status: incomplete. No usable concurrency limit has been measured yet.

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
  and validate `import openhands.agent_server.api` before packaging.
- Rivanna reports a default Apptainer session directory limit of 64 MiB. Its
  effect on these tasks' writable build trees has not yet been tested.
- No NOP/Oracle baseline has passed on these new task images. Therefore no
  concurrency or throughput claim is supported, including at concurrency 3.
- SSH connectivity to all three DNS addresses for the login service subsequently
  timed out. The terminal dependency fix was prepared locally but could not be
  uploaded. All submitted build/precheck jobs had already ended.

## Resume and reporting requirements

1. Restore SSH access and synchronize the prepared image fix.
2. Validate server import/startup, task working directory, NOP reward 0, and Oracle
   reward 1 for each selected task. Diagnose writable-layer failures separately.
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
