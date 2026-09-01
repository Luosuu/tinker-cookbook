Fix the following issue in the Kokkos repository.

Correct `Kokkos::reduction_identity` for integral reducers: the bitwise-AND (`BAnd`) identity must be `~Scalar{0}` (all bits set) for every integral scalar type, including `unsigned int` and signed variants; the incorrect value `0` breaks reducer initialization and empty-range `parallel_reduce`. Preserve the correct identities for bitwise OR (`BOr` → `0`), logical AND (`LAnd` → `1`/true), and logical OR (`LOr` → `0`/false), and leave `max`/`min` unchanged. The fix must apply consistently across all integral types so that reducing over zero iterations yields the proper neutral element for each reducer.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
