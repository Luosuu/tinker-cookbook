Fix the following issue in the Kokkos repository.

Add host-only functions Kokkos::nexttoward, Kokkos::nexttowardf, and Kokkos::nexttowardl. Provide overloads for (float, long double) returning float, (double, long double) returning double, (long double, long double) returning long double, and (integral, long double) returning double. In all overloads the second argument is long double. These are host-only and must not be available on device.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
