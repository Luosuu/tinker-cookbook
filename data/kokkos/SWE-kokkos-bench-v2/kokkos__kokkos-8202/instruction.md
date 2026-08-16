Fix the following issue in the Kokkos repository.

This enables passing extra integer arguments to the respective places in Kokkos View as the current Sacado partial specialization allows. We likely want to deprecate these code paths in a subsequent PR. For `shmem_size` and `required_allocation_size` we need to invent new interfaces however to do that (hence subsequent PR)

Specifically for construction for Sacado the following is equivalent now:

[implementation suggestion omitted]

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
