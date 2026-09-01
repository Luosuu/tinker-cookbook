Fix the following issue in the Kokkos repository.

Provide class-template argument deduction guides for Kokkos::TeamPolicy. Constructors with no execution-space argument, or with DefaultExecutionSpace, must deduce TeamPolicy<>. Constructors with an execution-space argument ES for which is_execution_space_v<ES> is true and ES differs from DefaultExecutionSpace must deduce TeamPolicy<ES>. Arguments that are only implicitly convertible to DefaultExecutionSpace but do not satisfy is_execution_space_v must result in TeamPolicy<>, not TeamPolicy<ArgumentType>. The implementation must compile under LLVM 12, LLVM 18, GCC 11, and GCC 14.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
