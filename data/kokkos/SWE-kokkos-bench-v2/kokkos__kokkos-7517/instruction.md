Fix the following issue in the Kokkos repository.

This fixes using DynRankView in scratch space. Two issues are fixed:

* `shmem_size` didn't report correct size.
* ctor taking scratch space called host only function.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
