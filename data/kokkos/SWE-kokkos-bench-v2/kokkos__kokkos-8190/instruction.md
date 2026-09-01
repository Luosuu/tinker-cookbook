Fix the following issue in the Kokkos repository.

Add `Kokkos::Experimental::ThenPolicy<WorkTag>` (default `WorkTag = void`; `WorkTag` must be empty or `void`) so graph nodes can receive an execution-policy work tag. Extend `GraphNodeRef::then` with overloads accepting `(label, policy, functor)`, `(policy, functor)`, `(label, functor)`, and `(label, exec, policy, functor)`, with existing overloads defaulting to `ThenPolicy<>`. Invoke the functor as `functor()` when the work tag is `void`, and as `functor(WorkTag{})` when it is a non-void empty type. Preserve full backward compatibility so all prior `then` usages compile unchanged.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
