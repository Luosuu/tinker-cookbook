Fix the following issue in the Kokkos repository.

The mdspan-based View implementation didn't check the extents when constructing a subview. This pull request creates a corresponding check and adds a test.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
