Fix the following issue in the Kokkos repository.

In the mdspan-based Kokkos::View implementation, constructing a subview via Kokkos::subview does not validate slice arguments against the source view extents. Add bounds checking that is active only when KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK is defined. For each dimension: an integer index must be less than the source extent; a std::pair<T,T> or Kokkos::pair<T,T> range must satisfy 0 <= first <= second <= extent; Kokkos::ALL is always valid. If any slice violates these rules, abort with a message containing 'Kokkos::subview bounds error'. When KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK is not enabled, behavior must remain unchanged.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
