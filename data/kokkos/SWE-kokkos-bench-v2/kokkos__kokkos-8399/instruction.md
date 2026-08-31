Fix the following issue in the Kokkos repository.

* Get rid of the GCC 10.3 workaround.
* Fix copy semantics (broken in https://github.com/kokkos/kokkos/pull/4348, giving a link time error instead of compile time)
* Modernize the implementation
* Add test coverage

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
