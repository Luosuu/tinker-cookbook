Fix the following issue in the Kokkos repository.

Added `Kokkos::Experimental::BadAllocation` as a typed exception thrown when memory allocation fails. Inherits from `std::runtime_error` for backwards compatibility. Makes programmatic OOM handling easier.

A corresponding typed-catch test is added to `TestViewBadAlloc`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
