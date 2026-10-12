Fix the following issue in the Kokkos repository.

In Kokkos::parallel_for, both overloads of Kokkos::parallel_scan, and Kokkos::parallel_reduce, shared-allocation tracking is manually disabled before copying the user functor into the internal closure and re-enabled afterward. This is not exception-safe across all backends: if the functor's copy constructor throws (e.g., std::bad_alloc), tracking remains disabled and corrupts view reference counting until the next parallel operation.

Replace all manual toggling with exception-safe mechanisms. Introduce SharedAllocationDisableTrackingGuard (an RAII guard) and construct_with_shared_allocation_tracking_disabled (a constructor helper) to manage tracking state safely around functor copies. Remove the old shared_allocation_tracking_disable() and shared_allocation_tracking_enable() functions. Apply these consistently to every affected parallel operation closure so that a thrown copy constructor always restores tracking, ensuring subsequent view copies update reference counts properly.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
