Fix the following issue in the Kokkos repository.

Remove the obsolete `impl/Kokkos_FixedBufferMemoryPool.hpp` header. Its `FixedBlockSizeMemoryPool` implementation is unused and has not been included by production code since the task-scheduler changes that made it redundant. Ensure no production reference to this header or type remains.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
