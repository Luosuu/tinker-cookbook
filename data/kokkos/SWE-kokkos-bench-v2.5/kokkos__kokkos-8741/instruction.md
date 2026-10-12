Fix the following issue in the Kokkos repository.

Implement remquo and its explicit-type variants (remquof, remquol) in the Kokkos mathematical functions API, along with generic arithmetic overloads that apply standard promotion rules (float/double, or long double when present). For inputs x and y and an int* quo parameter, each overload must store the truncated-toward-zero quotient of x/y in *quo and return the floating-point remainder with the same sign as x and magnitude strictly less than |y|. The functions must be callable in both host and device contexts, visible in the Kokkos namespace through the public module interface, and consistent with the behavior and promotion conventions of other mathematical functions listed for version 5.1. Provide all required signatures covering float, double, long double, and generic arithmetic pairs.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
