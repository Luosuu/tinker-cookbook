Fix the following issue in the Kokkos repository.

Adding tests that checks that non-defaulted views can only be constructed after Kokkos::initialize() was called and must be destructed before Kokkos::finalize().
Change the behavior from throwing a runtime error to aborting in the case of View creations.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
