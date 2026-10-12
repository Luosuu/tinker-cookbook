Fix the following issue in the Kokkos repository.

Make the two host-mirror aliases exposed by `Kokkos::View`, `HostMirror` and `host_mirror_type`, exactly the same type. Preserve the established `HostMirror` semantics: the mirror uses the view's non-const data type and array layout, `DefaultHostExecutionSpace` paired with the selected host-mirror memory space, and the existing hooks policy. Define `host_mirror_type` with those semantics and make `HostMirror` an alias of it so existing code using either spelling remains compatible.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
