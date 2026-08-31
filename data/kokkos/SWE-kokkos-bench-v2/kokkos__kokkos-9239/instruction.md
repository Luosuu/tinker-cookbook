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
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
