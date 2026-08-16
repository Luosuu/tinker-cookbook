Fix the following issue in the Kokkos repository.

There are 2 host mirror view types defined in `Kokkos::View`:
- `HostMirror` (`PascalCase`)
- `host_mirror_type` (`snake_case`)

This PR aligns both types so they are strictly equivalent (tested),

Note that the type defined by `HostMirror` is kept, the one defined previously by `host_mirror_type` disappears. This is because `HostMirror` is used all over the place in `Kokkos`, `host_mirror_type` was not.

This PR addresses #6996.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
