Fix the following issue in the Kokkos repository.

Added `Kokkos::Experimental::BadAllocation` as a typed exception thrown when memory allocation fails. Inherits from `std::runtime_error` for backwards compatibility. Makes programmatic OOM handling easier.

A corresponding typed-catch test is added to `TestViewBadAlloc`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
