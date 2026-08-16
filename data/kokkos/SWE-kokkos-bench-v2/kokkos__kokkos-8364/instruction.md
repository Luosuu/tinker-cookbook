Fix the following issue in the Kokkos repository.

This PR adds 2 new helpers for `Kokkos::Impl::type_list`:
1. `type_list_size_v` to get its size.
2. `type_list_contains_v` to know if it contains a type.

Related to:
- #8191

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
