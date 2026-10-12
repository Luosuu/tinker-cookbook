Fix the following issue in the Kokkos repository.

Fix LayoutRight handling in View and DynRankView, removing an erroneous conversion abort and ensuring correct layout propagation, with distinct legacy and new-implementation behavior governed by KOKKOS_ENABLE_IMPL_VIEW_LEGACY.

In non-legacy builds, LayoutRight must be equivalent to layout_right_padded: padding applies to the rightmost dimension, and stride is derived from the source mapping at rank-2 (falling back to 1 for rank 0 or 1). The abort that incorrectly prevents this mapping conversion in the new view path must be eliminated. In legacy builds, LayoutRight retains its original left-padded behavior with existing left-based stride rules.

DynRankView must unconditionally preserve LayoutRight layouts and mappings, so that constructing a DynRankView from a LayoutRight layout yields correct strides and padding—right-padded under the new implementation and left-padded under legacy—regardless of internal instantiation parameters. The behavior must be consistent for all ranks, including rank <= 1, without affecting other layout types.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
