Fix the following issue in the Kokkos repository.

Extracted from https://github.com/kokkos/kokkos/pull/7480/files#r1821082227. Basically, the specialization is missing to define a `sequential_host_init` alias. Let's see if we can just remove the specialization and if not we'll add the missing alias.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
