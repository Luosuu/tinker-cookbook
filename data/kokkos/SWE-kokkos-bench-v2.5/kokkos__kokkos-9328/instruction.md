Fix the following issue in the Kokkos repository.

Fix Kokkos::TeamPolicy’s templated converting constructor so it takes a const TeamPolicy<OtherProperties...>& rather than by value. When OtherProperties... is empty (<>), a by-value parameter becomes TeamPolicy(const TeamPolicy<> p), which conflicts with the copy constructor and fails with nvcc. The constructor must remain distinct, unambiguous, and valid for all property packs—including <>—without altering other behavior.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
