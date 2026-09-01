Fix the following issue in the Kokkos repository.

Introduce the public macros KOKKOS_FORCEINLINE_LAMBDA and KOKKOS_FORCEINLINE_CLASS_LAMBDA to specify that a host-device lambda expression should always be inlined. These macros must be defined after KOKKOS_LAMBDA, KOKKOS_CLASS_LAMBDA, and KOKKOS_IMPL_FORCEINLINE_ATTRIBUTE are available. In compiler modes that support lambda-expression front attributes (excluding C++20 and Clang in C++23 mode), KOKKOS_FORCEINLINE_LAMBDA must combine KOKKOS_LAMBDA with KOKKOS_IMPL_FORCEINLINE_ATTRIBUTE, and KOKKOS_FORCEINLINE_CLASS_LAMBDA must combine KOKKOS_CLASS_LAMBDA with the same attribute. In C++20 and in Clang’s C++23 mode, where such front placement is unsupported, both macros must fall back to the corresponding plain lambda macro without the force-inline attribute. The macros must be exposed through the public macro interface alongside the existing lambda macros.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
