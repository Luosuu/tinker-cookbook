Fix the following issue in the Kokkos repository.

Fix the graph API so `then_parallel_for` and `then_parallel_reduce` accept execution policies passed as lvalues, including stored `RangePolicy`, `MDRangePolicy`, and `TeamPolicy` objects and policies modified through methods such as `set_scratch_size`. Policy handling must use the underlying non-reference policy type while preserving normal forwarding behavior. Add the missing convenience overload `then_parallel_reduce(Label, Policy, Functor, ReturnType)` consistent with the labeled `then_parallel_for` interface. Also make `GraphNodeImpl` destructor overrides explicitly `noexcept` so graph nodes with complex functor types compile without an exception-specification mismatch. Existing rvalue-policy calls and TeamPolicy launch-bound properties must continue to work.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
