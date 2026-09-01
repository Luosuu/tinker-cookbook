Fix the following issue in the Kokkos repository.

Make NVHPC builds warning-clean when unreachable-code diagnostics are treated as errors. Add `KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_PUSH` and `KOKKOS_IMPL_DISABLE_UNREACHABLE_WARNINGS_POP`: under `__NVCOMPILER` they must suppress and then restore the compiler's unreachable-code and unreachable-initialization diagnostics, and on other compilers they must be harmless no-ops. Also ensure the mdspan padded-layout fallback returns used for NVCC or Intel missing-return warnings are not enabled for `__NVCOMPILER`, where they themselves trigger unreachable-code warnings.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
