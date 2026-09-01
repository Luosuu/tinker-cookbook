Fix the following issue in the Kokkos repository.

In Kokkos::Impl::get_ctest_gpu, the CTest GPU resource allocation parser, change every error path from throwing an exception to calling abort with the same formatted message. Cover all parsing failures—including invalid rank, missing or invalid group or device type, and missing or invalid ID strings—so the function terminates immediately with the original diagnostic instead of throwing std::runtime_error.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
