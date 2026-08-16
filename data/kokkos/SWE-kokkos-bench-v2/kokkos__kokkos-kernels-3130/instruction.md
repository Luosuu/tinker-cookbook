Fix the following issue in the Kokkos repository.

This PR introduces a new class for batched norm computation.
Based on the [older blas implementation](https://www.netlib.org/lapack/lapack-3.1.1/html/dznrm2.f.html), l2 norm can be computed without overflow/underflow.

- [x] Add a new class `ScaledL2`
- [x] Add an unit test to show that `ScaledL2` gives correct results without overflow while it gives the same results as `L2`

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
