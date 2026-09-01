Fix the following issue in the Kokkos repository.

Make `Kokkos::complex<RealType>` fully usable in constant expressions, matching C++14 `std::complex` behavior. The public API must support `constexpr` default construction, one-argument and two-argument constructors (`complex(const RealType&)` and `complex(const RealType&, const RealType&)`), copy and move construction and assignment, and scalar assignment (`operator=(const RealType&)`). All of these must be valid in `constexpr` contexts so instances can be initialized, assigned, copied, and moved inside `constexpr` functions and `static_assert` without requiring runtime evaluation.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
