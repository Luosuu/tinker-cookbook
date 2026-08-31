Fix the following issue in the Kokkos repository.

Implement `Kokkos::norm` both for `Kokkos::complex` and additional overloads for floating point and integer types.

Quick reference:
* https://en.cppreference.com/w/cpp/numeric/complex/norm.html

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
