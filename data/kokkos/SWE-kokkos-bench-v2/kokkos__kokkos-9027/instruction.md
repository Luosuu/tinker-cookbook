Fix the following issue in the Kokkos repository.

This PR is a pre-requisite for #9012 , whose graph tests depend on being able to pass lvalue execution policies to `then_parallel_for` and `then_parallel_reduce`.

## Summary

- Fix `then_parallel_for` and `then_parallel_reduce` in the graph API to accept lvalue execution policies (e.g., policies stored in variables or modified via `set_scratch_size`)
- Add a missing `(Label, Policy, Functor, ReturnType)` convenience overload for `then_parallel_reduce`, matching the existing pattern in `then_parallel_for`
- Fix `noexcept` mismatch in `GraphNodeImpl` destructor overrides that caused compilation failures with complex functor types (destructors that can throw)
- Minor expansion of test coverage, test a TeamPolicy with launch bounds

## Graph API fixes

The forwarding reference deduction in `then_parallel_for` and `then_parallel_reduce` (`Kokkos_GraphNode.hpp`) caused `Policy` to be deduced as a reference type when an lvalue was passed, making the `PolicyUpdate` constructor call ill-formed. Fixed by using `std::remove_cvref_t<Policy>`.

## Test plan

New tests:
- `lvalue_policies`: Lvalue RangePolicy, MDRangePolicy, TeamPolicy in graph nodes
- `lvalue_policies_reduce`: Lvalue RangePolicy with `then_parallel_reduce`
- `team_launch_bounds_in_graph`: TeamPolicy with `LaunchBounds` in graph nodes

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
