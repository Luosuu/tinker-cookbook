Fix the following issue in the Kokkos repository.

Add `KokkosBatched::Norm::ScaledL2` to the norm enumeration and update the `is_norm` trait so it is recognized. Extend `SerialNrm`, `TeamNrm`, and `TeamVectorNrm` to support `Norm::ScaledL2` for both real and complex scalar types under `Mode::Serial`, `Mode::Team`, and `Mode::TeamVector`. The computation must determine the L2 norm using a scaled accumulation that prevents intermediate overflow and underflow, yielding results mathematically equivalent to `Norm::L2` within floating-point tolerance. For inputs with magnitudes near the scalar type's maximum finite value, it must not overflow, unlike a direct square-sum L2 approach.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
