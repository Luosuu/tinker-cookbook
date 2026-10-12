Fix the following issue in the Kokkos repository.

Make the production code clean under clang-tidy's `bugprone-return-const-ref-from-parameter` check. Fix the identity functor used by STL-like inclusive and exclusive scan algorithms so it perfectly forwards its argument and no longer requires a value-type template parameter. In graph and ordinary parallel-reduce adapters, replace the obsolete conditional-selection helpers with a device-callable forwarding selection that preserves the chosen functor or reducer value category; remove the now-unused `Impl::ParallelReduceFunctorType` and `Impl::if_c` utilities. Where `ParallelReduceReturnValue<View>::return_value()` intentionally returns the supplied object by reference, retain the behavior and narrowly suppress the false positive.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
