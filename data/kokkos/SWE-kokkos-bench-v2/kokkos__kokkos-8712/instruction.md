Fix the following issue in the Kokkos repository.

Add Kokkos::isnormal to the Kokkos namespace, returning bool. For integral arguments return true iff the value is non-zero. For floating-point arguments (float, double, long double when enabled, and Kokkos::Experimental::half_t / bhalf_t) return true only when abs(x) is within [norm_min_v<T>, finite_max_v<T>] inclusive; return false for zero, infinity, NaN, and denormal/subnormal values. Provide half_t and bhalf_t overloads comparing against their respective norm_min_v and finite_max_v bounds, with a HIP NaN guard (x != x) in those overloads when HIP is enabled. Include the numeric traits header and export the symbol in the core module. Ensure correct backend behavior: provide direct template implementations for CUDA while supporting the standard unary-predicate path elsewhere. Guarantee decltype(isnormal(...)) is bool for all supported argument types.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
