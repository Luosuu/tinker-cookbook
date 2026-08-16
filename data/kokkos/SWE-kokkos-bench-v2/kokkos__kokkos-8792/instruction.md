Fix the following issue in the Kokkos repository.

This PR adds `noexcept` specifiers so that `Kokkos::View` satisfies `std::is_nothrow_move_constructible`.

The main motivation is that the `std::execution` sender/receiver framework (P2300) mandates receivers must be nothrow-movable.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
