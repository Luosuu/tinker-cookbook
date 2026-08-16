Fix the following issue in the Kokkos repository.

This makes sure that View::[type/const_type/non_const_type/host_mirror_type] are using mdspan-style args if the primary template used mdspan-style args.

I am leaving Uniform type alone - because there is an argument that they should all map to the same. Though we may want to reconsider which style they should use.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
