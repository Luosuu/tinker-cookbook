Fix the following issue in the Kokkos repository.

This enables passing extra integer arguments to the respective places in Kokkos View as the current Sacado partial specialization allows. We likely want to deprecate these code paths in a subsequent PR. For `shmem_size` and `required_allocation_size` we need to invent new interfaces however to do that (hence subsequent PR)

Specifically for construction for Sacado the following is equivalent now:

[implementation suggestion omitted]

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
