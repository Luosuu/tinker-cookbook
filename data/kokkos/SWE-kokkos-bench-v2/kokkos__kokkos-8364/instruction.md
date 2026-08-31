Fix the following issue in the Kokkos repository.

This PR adds 2 new helpers for `Kokkos::Impl::type_list`:
1. `type_list_size_v` to get its size.
2. `type_list_contains_v` to know if it contains a type.

Related to:
- #8191

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
