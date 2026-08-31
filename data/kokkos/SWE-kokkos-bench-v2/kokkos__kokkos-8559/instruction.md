Fix the following issue in the Kokkos repository.

The scratch view constructor didn't take into account the view customization. I am only allowing it for the ctor which takes integer args, not the one which takes a layout. We should eventually add ctors which take mapping/accessor.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
