Fix the following issue in the Kokkos repository.

Extend Kokkos::Random_XorShift64_Pool and Kokkos::Random_XorShift1024_Pool to support construction with an explicit execution_space instance, enabling asynchronous initialization on that space and correct device selection when multiple instances exist. For each pool, add the public constructors Pool(const execution_space&, uint64_t seed) and Pool(const execution_space&, uint64_t seed, uint64_t num_states). These overloads must initialize the pool on the provided space without fencing (non-blocking). The existing constructors taking only seed, or seed and num_states, must continue to use execution_space() and perform blocking initialization (fence after init). Expose the public overload void init(uint64_t seed, uint64_t num_states), which uses the default execution space and fences. The num_states parameter must be uint64_t consistently across all affected overloads. The pool’s internal state must be allocated and initialized on the execution space given at construction. Keep all existing APIs and behaviors intact.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
