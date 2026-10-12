Fix the following issue in the Kokkos repository.

Modify the Kokkos::View contract so that any non-default construction initializing underlying data requires an active default execution space initialized via Kokkos::initialize; if uninitialized, the operation must abort instead of throwing a runtime exception. This applies even to zero-size allocations. On CUDA, HIP, SYCL, or OpenACC backends, the failure may manifest as the backend's own execution-space initialization error. Additionally, destroying an allocating view after Kokkos::finalize has been called must abort, enforcing that allocations do not outlive the initialized scope. Default-constructed views and assignments performed entirely within a valid initialize/finalize window must remain valid. The change affects public view lifetime and initialization ordering, preserving compatibility for correctly scoped usage.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
