Fix the following issue in the Kokkos repository.

Implement `Kokkos::norm` overloads for `Kokkos::complex<RealType>` and scalar arithmetic types (integral and floating-point). For complex values, return the squared magnitude (`real()*real() + imag()*imag()`) with a result type matching the real component type. For arithmetic values, return `x * x` with promoted return types: integral arguments yield `double`, while floating-point arguments preserve their original type (e.g., `float` yields `float`). Both overloads must be declared `constexpr` and annotated with `KOKKOS_INLINE_FUNCTION`. Ensure `norm` is exported through the Kokkos module interface as part of the public API.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
