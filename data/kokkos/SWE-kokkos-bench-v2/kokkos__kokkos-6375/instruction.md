Fix the following issue in the Kokkos repository.

Add the public utility `Kokkos::to_array`, analogous to `std::to_array`, that produces `Kokkos::Array<std::remove_cv_t<T>, N>` from built-in arrays and braced-init-lists. Provide `constexpr KOKKOS_FUNCTION` overloads taking `T (&)[N]` and `T (&&)[N]`, deducing element type and length. Also support braced-init-list temporary arrays with deduced size and type, and allow explicit element-type specification with implicit conversion (for example, `Kokkos::to_array<long>({0, 1, 3})`). The result must have cv-qualifiers removed from the element type.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
