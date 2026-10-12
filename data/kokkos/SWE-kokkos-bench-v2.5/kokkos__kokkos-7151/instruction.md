Fix the following issue in the Kokkos repository.

Make the ChunkSize(int) constructor explicit so that int is not implicitly converted to ChunkSize. This eliminates the unintended RangePolicy constructor behavior of accepting any type convertible to ChunkSize as its final argument and implicitly constructing ChunkSize. Explicit ChunkSize(int) construction must remain valid. When KOKKOS_ENABLE_DEPRECATED_CODE_4 is enabled, preserve implicit int-to-ChunkSize conversion for backward compatibility so RangePolicy continues to allow it; otherwise RangePolicy must require an explicit ChunkSize where one is expected.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
