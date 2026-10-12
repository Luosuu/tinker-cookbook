Fix the following issue in the Kokkos repository.

For the pointer-type specialization of ViewCtorProp (T*), define the missing static constexpr bool member sequential_host_init with value false. Retain this specialization; do not remove it. The constructor-property interface for pointer-backed views must remain complete and expose sequential_host_init alongside its existing members.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
