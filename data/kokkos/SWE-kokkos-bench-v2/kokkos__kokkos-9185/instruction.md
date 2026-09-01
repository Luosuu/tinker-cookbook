Fix the following issue in the Kokkos repository.

Add one-argument overloads of `create_mirror_view_and_copy` that accept a source view (`View`, and analogous public forms for `DynamicView`, `DynRankView`, and `OffsetView` where applicable). Each overload must produce a mirror view residing in the source view's designated host-mirror memory space—not hard-coded to `Kokkos::HostSpace`—and must copy the source data into it. This fixes compilation errors such as static_assert type mismatches on unified-memory architectures (e.g., MI300A with `KOKKOS_ARCH_AMD_GFX942_APU=ON`), where `HIPSpace` is host-accessible and the correct mirror space differs from `HostSpace`. The overloads must work correctly on both unified-memory and traditional discrete-memory architectures. The existing two-argument `create_mirror_view_and_copy(Space, View)` overloads must remain fully available and unchanged.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
