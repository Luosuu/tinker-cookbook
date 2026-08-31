Fix the following issue in the Kokkos repository.

`Kokkos::complex` should have `constexpr` constructor as well

If I'm not mistaken, according to https://en.cppreference.com/w/cpp/numeric/complex/complex there are several constructors of `std::complex` that are declared `constexpr` already in the `c++14` standard.

Looking at
[source location omitted]
it seems it is not the case for `Kokkos::complex`, and `constexpr` needs to be explicitly added.

Here is a small code to test `constexpr`-ness:
[implementation suggestion omitted]

This should show (at least for `g++ 11.3.0`):
[implementation suggestion omitted]

If you agree on the changes, I can open a PR.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
