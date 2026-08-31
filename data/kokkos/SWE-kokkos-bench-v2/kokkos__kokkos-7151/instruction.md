Fix the following issue in the Kokkos repository.

Release 4.3 unintentionally introduced the ability for `RangePolicy` constructor to take as it's final argument any type convertible to `ChunkSize` and implicitly construct a `ChunkSize` object for the input. Remove that by making the `ChunkSize(int)` constructor explicit.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
