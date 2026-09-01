Fix the following issue in the Kokkos repository.

RangePolicy-based overloads of Kokkos::parallel_for, Kokkos::parallel_reduce, and Kokkos::parallel_scan—including those taking a size_t work count—currently behave inconsistently when called before Kokkos::initialize or after Kokkos::finalize. They must consistently abort via Kokkos::abort in all such cases. The abort message must identify the called function, whether it was invoked before initialization or after finalization, the call label (or absence thereof), and the RangePolicy. This must hold for empty ranges and a work count of zero. Limit the change to range-policy variants; exclude MDRangePolicy and TeamPolicy overloads.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
