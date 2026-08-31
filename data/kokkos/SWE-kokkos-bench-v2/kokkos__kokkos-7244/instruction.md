Fix the following issue in the Kokkos repository.

There are 2 host mirror view types defined in `Kokkos::View`:
- `HostMirror` (`PascalCase`)
- `host_mirror_type` (`snake_case`)

This PR aligns both types so they are strictly equivalent (tested),

Note that the type defined by `HostMirror` is kept, the one defined previously by `host_mirror_type` disappears. This is because `HostMirror` is used all over the place in `Kokkos`, `host_mirror_type` was not.

This PR addresses #6996.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
