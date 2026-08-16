Fix the following issue in the Kokkos repository.

This PR adds a missing `constexpr` for `Kokkos::conj`, and for the `real` and `imag` free functions.

Quick reference:
* https://en.cppreference.com/w/cpp/numeric/complex/conj

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
