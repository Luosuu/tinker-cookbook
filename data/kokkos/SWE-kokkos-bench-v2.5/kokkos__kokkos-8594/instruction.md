Fix the following issue in the Kokkos repository.

In both public overloads of `Kokkos::Experimental::contribute`—with and without an explicit execution-space argument—remove the unnecessary non-const lvalue requirement on the destination `View`. Accept the destination as `View<...> const&` so temporary Views, such as inline subviews, can be passed while preserving existing behavior for lvalue destinations.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
