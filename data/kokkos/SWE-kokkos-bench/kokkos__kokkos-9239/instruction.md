Fix the following issue in the Kokkos repository.

Introduce macros to specify to always inline a host device lambda expressions, as proposed in #9212

One thing that I had not realized before writing the test was
[implementation suggestion omitted]

### Related issues / PRs

Superseding #9212 

### Changelog Entry

* Introduce `KOKKOS_[CLASS_]FORCEINLINE_LAMBDA` macros

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
