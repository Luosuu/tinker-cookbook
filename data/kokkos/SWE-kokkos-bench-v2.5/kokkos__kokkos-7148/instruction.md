Fix the following issue in the Kokkos repository.

Add constexpr hidden-friend == and != operators to Kokkos::Array<T, N> for all dimensions N, including N==0. For N>0 perform element-wise comparison; for N==0, == returns true and != returns false. These operators apply only between arrays with identical element type T and size N. Do not provide them for deprecated Kokkos::Array forms.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
