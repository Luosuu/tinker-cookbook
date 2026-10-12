Fix the following issue in the Kokkos repository.

Introduce the compile-time trait `Kokkos::Experimental::StaticBatchSize<N>` (`N > 0`) with `static constexpr unsigned int batch_size = N`. It must be usable as an optional `RangePolicy` template parameter (e.g., `RangePolicy<ExecSpace, WorkTag, StaticBatchSize<N>>`); when omitted, the default must be `StaticBatchSize<1>` so existing behavior is unchanged. When applied via `RangePolicy` to `parallel_for` and `parallel_reduce`, the loop range must be processed in batches of size `batch_size`, ensuring no iteration runs outside the declared range bounds. This must hold for ranges exactly divisible by the batch size, ranges with a remainder, and ranges smaller than the batch size. The contract applies to both `parallel_for` and `parallel_reduce`; no other policy behavior should change.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
