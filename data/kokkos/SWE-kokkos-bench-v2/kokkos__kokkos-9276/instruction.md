Fix the following issue in the Kokkos repository.

This makes sure that View::[type/const_type/non_const_type/host_mirror_type] are using mdspan-style args if the primary template used mdspan-style args.

I am leaving Uniform type alone - because there is an argument that they should all map to the same. Though we may want to reconsider which style they should use.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
