Fix the following issue in the Kokkos repository.

`team_scratch_size(int)` and `thread_scratch_size(int)` are currently only implemented for the CUDA, HIP and SYCL backends, this PR adds the function for the remaining backends (host backends and OpenACC) and adds a test for these two functions.

### Related issues / PRs

### Changelog Entry

Add support for querying the team and thread level scratch size on all backends

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
