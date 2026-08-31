Fix the following issue in the Kokkos repository.

When `OtherProperties...` is `<>`, nvcc saw it as `TeamPolicy(const TeamPolicy<> p)`, which is not a valid copy-ctor. For some reason my Serial build was happy with that, but `nvcc` was not.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
