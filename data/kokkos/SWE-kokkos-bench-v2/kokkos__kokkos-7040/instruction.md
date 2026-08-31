Fix the following issue in the Kokkos repository.

Following the model in [P2819R2](https://wg21..link/P2819R2) (voted into C++26 on 2023-11), this adds structured binding support for Kokkos::complex.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
