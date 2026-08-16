Fix the following issue in the Kokkos repository.

Adding tests that checks that non-defaulted views can only be constructed after Kokkos::initialize() was called and must be destructed before Kokkos::finalize().
Change the behavior from throwing a runtime error to aborting in the case of View creations.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
