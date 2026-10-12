Fix the following issue in the Kokkos repository.

In the `Kokkos::Impl` namespace, add three `constexpr` variable-template helpers for `Kokkos::Impl::type_list`: `type_list_size_v` (number of types, `0` if empty), `type_list_contains_v` (whether a specified type is present, `false` if empty), and `type_list_any_v` (whether any element satisfies a given unary predicate, `false` if empty). They must be usable in constant expressions and handle empty lists correctly.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
