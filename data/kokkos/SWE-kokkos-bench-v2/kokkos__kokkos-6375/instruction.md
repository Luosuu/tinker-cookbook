Fix the following issue in the Kokkos repository.

(Part of issue #6355, intended to be applied after #6372, as that contains the initial unit tests for `Kokkos::array`.)

Adding `Kokkos::to_Array` for `Kokkos::Array<T, N, void>`, analogous to `std::to_array` for `std::array`, as well as unit tests.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
