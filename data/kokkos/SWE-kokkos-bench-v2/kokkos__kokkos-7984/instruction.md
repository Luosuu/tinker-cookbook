Fix the following issue in the Kokkos repository.

We didn't test v.stride(r) with r>= rank, we didn't test stride() for layout_stride at all, and we didn't test that the last value in the array overload of stride() is correct.

I added the tests and made the new view implementation match the behavior of the legacy implementation. 

Note: our documentation says you can't use r>=rank for stride(r): https://kokkos.org/kokkos-core-wiki/API/core/view/view.html#_CPPv4I0ENK6strideE6size_tRK5iType.

I reintroduced this behavior under deprecation.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
