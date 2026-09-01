Fix the following issue in the Kokkos repository.

Restore `constexpr` `kokkos_swap` support for `Kokkos::Array<T, N>` in namespace `Kokkos`, including `N == 0`, matching `std::swap` behavior for `std::array`. For non-empty arrays, the overload must participate only when `T` is swappable, be `noexcept` exactly when swapping `T` is nothrow, and exchange elements through unqualified `kokkos_swap` calls so user-defined overloads are found by argument-dependent lookup. For zero-sized arrays, provide a no-op `noexcept` overload.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
