Fix the following issue in the Kokkos repository.

The SELL sparse matrix format is helpful when a matrix has a well balanced number of non-zeros per row and can leverage that structure to perform faster matrix-vector operations.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
