Fix the following issue in the Kokkos repository.

This PR aims at adding a batched interface for [rotg](https://www.netlib.org/lapack/explore-html/d7/dc5/group__rotg_gaafa91c51f75df6c3f2182032a221c2db.html#gaafa91c51f75df6c3f2182032a221c2db). 
Discussing points

1. Should we update the `b` value? It is updated only by (s/d)rotg but not with (c/z)rotg. It is unclear this value can be used in reality. 
2. Should we manage more carefully the cases for extremely small `a` and `b`?  This kernel can overflow in the current implementation.

- [x] Refactor `rotg_impl` under blas
- [x] Add batched interface `Rotg`. `SerialRotg` and `TeamRotg` are not added because this kernel works only on scalars without any parallelization
- [x] Tests introduced under batched (under blas tests are commented out for some reason)

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
