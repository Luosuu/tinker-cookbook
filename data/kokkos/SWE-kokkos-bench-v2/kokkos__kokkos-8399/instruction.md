Fix the following issue in the Kokkos repository.

* Get rid of the GCC 10.3 workaround.
* Fix copy semantics (broken in https://github.com/kokkos/kokkos/pull/4348, giving a link time error instead of compile time)
* Modernize the implementation
* Add test coverage

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
