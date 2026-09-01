Fix the following issue in the Kokkos repository.

Make Kokkos::UnorderedMap work with SequentialHostInit. Constructing an UnorderedMap whose allocation properties include SequentialHostInit—commonly via Kokkos::view_alloc(Kokkos::SequentialHostInit, ...)—currently fails because the implementation unconditionally appends WithoutInitializing to the internal properties, which conflicts with SequentialHostInit. This is especially needed when value_type is Kokkos::View, e.g.:

using value_type = Kokkos::View<size_t*, Kokkos::HostSpace>;
using map_type = Kokkos::UnorderedMap<int, value_type, Kokkos::HostSpace>;
map_type map(Kokkos::view_alloc(Kokkos::SequentialHostInit, "label"), 150);

The map must compile and operate correctly for construction with a capacity hint, insertion, copy construction, rehash, and assignment when SequentialHostInit is present. In those cases WithoutInitializing must not be added; when SequentialHostInit is absent, internal arrays should remain uninitialized as before. Size, capacity, and allocation state must remain correct in all paths.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
