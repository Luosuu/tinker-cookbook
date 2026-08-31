Fix the following issue in the Kokkos repository.

This PR aims at adding a batched interface for [rotmg](https://www.netlib.org/lapack/explore-html/d3/dd5/group__rotmg_gaebf62f1c90f0829a0a762a7f8918213f.html#gaebf62f1c90f0829a0a762a7f8918213f). 

- [x] Refactor `rotmg_impl` under blas. Fixed the behavior for `d1 < 0`.
- [x] Add batched interface `Rotmg`. `SerialRotmg` and `TeamRotmg` are not added because this kernel works only on scalars without any parallelization
- [x] Tests introduced under batched. Some corner case (`d1 < 0`) is not covered

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
