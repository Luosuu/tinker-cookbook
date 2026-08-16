Fix the following issue in the Kokkos repository.

Proposal: add begin/end to Kokkos::Array

Currently, `Kokkos::Array` does not have `begin()` and `end()` methods which are essential for C++ standard algorithm.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
