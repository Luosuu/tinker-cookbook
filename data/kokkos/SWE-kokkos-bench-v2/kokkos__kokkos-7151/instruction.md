Fix the following issue in the Kokkos repository.

Release 4.3 unintentionally introduced the ability for `RangePolicy` constructor to take as it's final argument any type convertible to `ChunkSize` and implicitly construct a `ChunkSize` object for the input. Remove that by making the `ChunkSize(int)` constructor explicit.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
