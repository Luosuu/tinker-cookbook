Fix the following issue in the Kokkos repository.

TeamPolicy exposes team_scratch_size(int) and thread_scratch_size(int) to retrieve the per-level scratch size (size_t) previously set via set_scratch_size. These query methods are currently available only on CUDA, HIP, and SYCL backends. Implement them for all remaining backends—Serial, Threads, OpenMP, OpenACC, and HPX—so that each returns the corresponding configured team-level or thread-level scratch size. Behavior must match the existing backend implementations and remain compatible with host and accelerator execution spaces.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
