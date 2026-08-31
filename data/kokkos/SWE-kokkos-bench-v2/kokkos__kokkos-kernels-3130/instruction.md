Fix the following issue in the Kokkos repository.

This PR introduces a new class for batched norm computation.
Based on the [older blas implementation](https://www.netlib.org/lapack/lapack-3.1.1/html/dznrm2.f.html), l2 norm can be computed without overflow/underflow.

- [x] Add a new class `ScaledL2`
- [x] Add an unit test to show that `ScaledL2` gives correct results without overflow while it gives the same results as `L2`

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
