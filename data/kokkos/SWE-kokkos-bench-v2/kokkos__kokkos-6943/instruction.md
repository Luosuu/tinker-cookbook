Fix the following issue in the Kokkos repository.

Specializing the swap algorithm for Kokkos arrays was initially proposed in #6697 but we dropped it to focus on the Kokkos swap ADL ordeal. Somehow we overlooked a stray <Kokkos_Swap.hpp> header include in the Kokkos::Array header file.  This PR reintroduce a
`Kokkos::kokkos_swap(Kokkos::Array)` specialization, following closely what the standard library does for `std::swap(std::array)`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
