Fix the following issue in the Kokkos repository.

* Fix identity operator in stl-like parallel numerics algorithms (used in prefix sums)
* Silent warning in `ParallelReduceReturnValue<View>::return_value()`
  * Current use is fine and I would want to get rid of it in a larger-scale refactoring
* Fix reducer initialization selecting either the functor or the "return" value in `[then_}parallel_reduce()`
  * The "forwarding switch" functionality is duplicated but it is unclear to me we need it anywhere else and I wasn't sure where we'd want it anyway
* Remove (unused) `Impl::ParallelReduceFunctorType` class template
* Remove `Impl::if_c` "trait"
* Fix half type test

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
