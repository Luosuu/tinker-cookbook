Fix the following issue in the Kokkos repository.

Ensure kokkos_malloc (all overloads), kokkos_realloc, and kokkos_free abort when called before Kokkos::initialize() or after Kokkos::finalize(), consistent with the existing kokkos_free behavior. The state check must occur before any pointer dereference or internal tracking access, so invalid or fake pointers still produce the correct initialization-state error rather than undefined behavior. These guards must be disabled when KOKKOS_ENABLE_THREADS is defined, since the Threads backend calls kokkos_malloc during its own initialization. Apply uniformly to both the before-initialize and after-finalize cases across all three APIs.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
