Fix the following issue in the Kokkos repository.

Modernize the Kokkos bit-manipulation API by replacing SFINAE with C++20 concepts and fixing the bit_width return type.

- Define a StandardUnsignedInteger concept matching the standard unsigned integer types.
- Constrain the bit-manipulation overloads—including byteswap (with std::integral), countl_zero, countl_one, countr_zero, countr_one, popcount, has_single_bit, bit_ceil, bit_floor, bit_width, rotl, rotr, and their builtin/experimental counterparts—with this concept (or std::integral for byteswap).
- Constrain bit_cast with a requires clause that sizeof(To) == sizeof(From) and both types are trivially copyable.
- Change the return type of bit_width and bit_width_builtin from the template parameter to int; keep all other return types unchanged.
- Preserve existing constexpr, noexcept, KOKKOS_FUNCTION, and [[nodiscard]] annotations. Include the standard concepts header. Observable behavior is identical except for the concept constraints and the corrected int return type for bit_width.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
