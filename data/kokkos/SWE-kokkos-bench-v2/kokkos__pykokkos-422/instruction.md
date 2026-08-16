Fix the following issue in the Kokkos repository.

This stems form an issue where a PyKokkos kernel was passed a non-contiguous numpy array, leading to a segfault. This should be caught on the PyKokkos side.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
