Fix the following issue in the Kokkos repository.

This supercedes #6599 (it incorporates the run time tests that @maartenarnst wrote there).

Adds hidden friends `==` and `!=` to `Kokkos::Array<T, N>` for all `N`, including 0.

Deliberately does not add them to the deprecated forms of `Kokkos::Array`.

Adds a compile time test for equality comparable.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
