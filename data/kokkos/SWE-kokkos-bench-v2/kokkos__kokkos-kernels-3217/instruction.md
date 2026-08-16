Fix the following issue in the Kokkos repository.

This PR aims at supporting [ icmax1](https://www.netlib.org/lapack/explore-html/d6/dde/group__imax1_gad7dd50548fcf1917d4bbf016742b90d6.html#gad7dd50548fcf1917d4bbf016742b90d6).

- [x] Add new LInf norm computation. Align the definition of Linf norm with Blas
- [x]  Add unit-tests for this
- [x] Update documentation

See also #3150

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
