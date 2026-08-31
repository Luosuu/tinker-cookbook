Fix the following issue in the Kokkos repository.

(Part of issue #6355, intended to be applied after #6372, as that contains the initial unit tests for `Kokkos::array`.)

Adding `Kokkos::to_Array` for `Kokkos::Array<T, N, void>`, analogous to `std::to_array` for `std::array`, as well as unit tests.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
