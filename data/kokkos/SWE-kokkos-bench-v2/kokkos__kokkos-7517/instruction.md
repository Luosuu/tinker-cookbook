Fix the following issue in the Kokkos repository.

Fix DynRankView for team scratch-space usage. DynRankView is rank-dynamic (up to 7 dimensions, backed by a rank-7 underlying view) and has two scratch-related defects.

1. DynRankView::shmem_size returns an incorrect byte size. It must report the scratch bytes required by the underlying fixed-rank view given the current dynamic dimensions, using the appropriate default value for any unspecified (KOKKOS_INVALID_INDEX) dimensions, instead of an erroneous manual product.

2. The DynRankView constructors accepting scratch_memory_space—both layout-only and dimension-list overloads—invoke host-only operations and are invalid in device/parallel contexts. They must initialize the underlying view directly from the provided scratch space without relying on host-only helpers.

Preserve full rank-dynamic compatibility and ensure correct behavior when dimensions are partially specified.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
