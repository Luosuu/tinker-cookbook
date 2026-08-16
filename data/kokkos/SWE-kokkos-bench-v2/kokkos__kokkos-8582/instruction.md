Fix the following issue in the Kokkos repository.

In `Kokkos::Array`, added `noexcept` to `data()`, `empty()`, `size()`, `max_size()` to make it consistent with `std::array`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
