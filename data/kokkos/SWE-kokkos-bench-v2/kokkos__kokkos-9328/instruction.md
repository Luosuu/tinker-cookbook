Fix the following issue in the Kokkos repository.

When `OtherProperties...` is `<>`, nvcc saw it as `TeamPolicy(const TeamPolicy<> p)`, which is not a valid copy-ctor. For some reason my Serial build was happy with that, but `nvcc` was not.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
