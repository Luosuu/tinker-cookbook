Fix the following issue in the Kokkos repository.

Modernize `Kokkos::Timer` and remove the obsolete GCC 10.3 workaround and `gettimeofday` fallback. Use `std::chrono::high_resolution_clock` with a stored clock time point. Preserve the public default constructor, `reset()`, and const `seconds()` returning `double`. Explicitly delete copy construction, copy assignment, move construction, and move assignment so misuse is rejected at compile time rather than failing at link time. No compiler-specific timing path should remain.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
