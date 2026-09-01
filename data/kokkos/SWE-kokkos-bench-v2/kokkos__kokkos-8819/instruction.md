Fix the following issue in the Kokkos repository.

Implement `Kokkos::fpclassify` in the `Kokkos` namespace as a public function returning `int`, with overloads for integral types, `float`, `double`, `long double` (only when `MATHEMATICAL_FUNCTIONS_HAVE_LONG_DOUBLE_OVERLOADS` is defined), `Kokkos::Experimental::half_t`, and `Kokkos::Experimental::bhalf_t`. It must return `FP_NAN` when `x != x`; `FP_ZERO` for zero and negative zero; `FP_SUBNORMAL` when the absolute value is below the type's normal minimum; `FP_INFINITE` when the absolute value equals infinity; and `FP_NORMAL` otherwise. For integral inputs, return `FP_ZERO` for `0` and `FP_NORMAL` for all other values. The function must work correctly on CUDA, SYCL, and NVHPC without relying on a missing or subnormal-incorrect platform `fpclassify`. When `__FINITE_MATH_ONLY__` is defined, correct handling of NaN, infinite, and subnormal cases is not required. A naive, unoptimized implementation is acceptable.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
