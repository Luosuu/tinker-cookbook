Fix the following issue in the Kokkos repository.

Fix DynRankView layout initialization for Kokkos::LayoutRight when converting from a Kokkos::View. The layout struct’s stride member is computed incorrectly, while the view mapping must remain untouched and LayoutLeft behavior is unchanged. The bug is rank-sensitive: lower ranks such as rank-2 fail, whereas others like rank-3 may appear correct. For LayoutRight, DynRankView::layout() must report correct dimensions and must set stride to KOKKOS_INVALID_INDEX when the last dimension equals the source layout stride; otherwise the original stride value must be preserved. Ensure correct behavior across all supported ranks without altering the mapping.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
