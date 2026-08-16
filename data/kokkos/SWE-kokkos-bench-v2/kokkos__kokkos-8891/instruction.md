Fix the following issue in the Kokkos repository.

This PR adds [nexttoward](https://en.cppreference.com/w/cpp/numeric/math/nextafter.html).
This function has following overloads

[implementation suggestion omitted]

As the second argument is always `long double`, this is a host-only function.
In addition, some tests are skipped if `finite-math` is enabled

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
