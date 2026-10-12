Fix the following issue in the Kokkos repository.

Implement the missing Kokkos floating-point decomposition functions Kokkos::modf, Kokkos::modff, and Kokkos::modfl to match std::modf semantics. Provide overloads for float, double, and long double, each taking a value and a pointer to the corresponding floating-point type that receives the signed integral part; return the signed fractional part. Include an overload that promotes an integral argument to double. All variants must be device-callable and exported in the Kokkos core public interface. The other listed items—frexp, ldexp, scalbn, scalbln, and nexttoward—are excluded from this task unless explicitly required separately.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
