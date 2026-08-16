Fix the following issue in the Kokkos repository.

This is enabled by introducing `begin()` and `end()` free functions that take const and non-const  `Array` and return pointers to elements.

Whether or not to introduce them as member function can be decided elsewhere.
We would still need the free functions as the `std::` functions would lack the `__host__ __device__` annotations.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
