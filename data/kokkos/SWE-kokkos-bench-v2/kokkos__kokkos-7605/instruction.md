Fix the following issue in the Kokkos repository.

Extracted from https://github.com/kokkos/kokkos/pull/7480/files#r1821082227. Basically, the specialization is missing to define a `sequential_host_init` alias. Let's see if we can just remove the specialization and if not we'll add the missing alias.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
