Fix the following issue in the Kokkos repository.

This PR relaxes the constraints on `Kokkos::Array` `operator[]` to allow custom type indexers.

The current constraint, which requires `std::is_integral`, prevents the use of convertible types in its subscript operators, unlike `std::array`.

This change aligns with `Kokkos::View` interface, which previously also mandated `std::is_integral` in its legacy implementation:
[source location omitted]

but was updated to:
[source location omitted]

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
