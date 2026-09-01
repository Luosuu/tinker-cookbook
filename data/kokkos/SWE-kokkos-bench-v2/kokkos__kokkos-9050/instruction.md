Fix the following issue in the Kokkos repository.

RangePolicy provides class-template argument deduction guides for constructors that do not take an execution-space argument: RangePolicy(), RangePolicy(int64_t, int64_t), and RangePolicy(int64_t, int64_t, ChunkSize const&). These guides currently deduce RangePolicy<DefaultExecutionSpace>, but specifying the default execution space explicitly is unnecessary: the empty specialization resolves to the default execution space automatically and selects the correct specialization. Update only these three guides to deduce RangePolicy<>. Do not modify the deduction guide that accepts an explicit DefaultExecutionSpace& argument. The required observable behavior is that CTAD for the affected constructor forms yields RangePolicy<>.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
