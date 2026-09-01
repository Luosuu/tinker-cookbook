Fix the following issue in the Kokkos repository.

Enforce argument validation in Kokkos::TeamPolicy constructors for both execution-space and default-space overloads. For explicit integer arguments, require league_size to be non-negative, team_size and vector_length to be at least one, and—unless KOKKOS_ENABLE_DEPRECATED_CODE_5 is defined—vector_length to not exceed TeamPolicy::vector_length_max(). Arguments specified as Kokkos::AUTO must be excluded from numeric bounds checks. Any violation must trigger Kokkos::abort.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
