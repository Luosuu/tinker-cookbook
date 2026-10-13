# CUDA lane on Rivanna: environment and timing basis

Probed 2026-10-13 on `gpu` partition nodes with one NVIDIA RTX A6000 (compute capability 8.6,
driver 595.71.05), 8 CPUs, inside Apptainer with `--nv`. Each probe fetched the candidate's
base commit, configured `-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON
-DKokkos_ENABLE_TESTS=ON` with `bin/nvcc_wrapper`, built the core CUDA unit-test target with
`--parallel 8`, and ran its first 20 tests.

| Candidate | Era | Image | Arch | Target | Build (s) | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| kokkos-2644 (2020-01) | cpp14 | `nvidia/cuda:11.8.0-devel-ubuntu20.04` (GCC 9.4) | VOLTA70 | `KokkosCore_UnitTest_Cuda` | 270 | 20/20 passed |
| kokkos-3522 (2021) | cpp14 | same | AMPERE80 | `KokkosCore_UnitTest_Cuda1` | 189 | 20/20 passed |
| kokkos-7388 (2024) | cpp17 | `nvidia/cuda:12.2.2-devel-ubuntu22.04` (GCC 11.4) | AMPERE80 | `Kokkos_CoreUnitTest_Cuda1` | 361 | 19/19 passed |
| kokkos-9139 (2026) | cpp20 | `nvidia/cuda:12.8.1-devel-ubuntu24.04` (GCC 13.3) | AMPERE80 | `Kokkos_CoreUnitTest_Cuda1` | 522 | not run |

Kokkos before 3.1 has no `Kokkos_ARCH_AMPERE80` (configure stops with "to enable an NVIDIA GPU
architecture"); its VOLTA70 build embeds PTX that the driver compiles for Ampere at load.
AMPERE80 binaries run on every A100, A6000 and A40. They do not run on the RTX PRO 6000
(compute capability 12.0): Kokkos aborts at initialization with "running kernels compiled for
compute capability 8.0 on device with compute capability 12.0 is not supported by CUDA".

The GPU execution tier in `dataset/qualification.py` therefore allows 1800 s per target build
(over 3x the slowest probe at the same parallelism) and 8 build jobs, and the Rivanna bridge
requests `gpu:a6000:1` with 8 CPUs and 64 GB (16 GB and 32 GB ran out of memory while
squashing CUDA images).
