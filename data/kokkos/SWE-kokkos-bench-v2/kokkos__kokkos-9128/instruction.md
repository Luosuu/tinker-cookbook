Fix the following issue in the Kokkos repository.

<!-- Provide a short summary of the changes in this pull request. -->
Calling `Kokkos::resize(exec_space{}, v, new_layout)` with an explicit `ExecutionSpace` arg causes a compile error due to ambiguous overload. Two overload :
1. The generic 
	[implementation suggestion omitted]
 3. The explicit 
 	[implementation suggestion omitted]

Since `ExecutionSpace` can explicitly wrapped in a `ViewCtorProp`, both overload are viable candidate.
Fix: Remove redundant resize(const ExecutionSpace& , ...) overload. The generic overload already handles the case correctly by wrapping via Kokkos::view_alloc().
Testing: Added TestCopyViewsBugs.hpp with tests that verify Kokkos::resize(exec_space{}, v, new_layout) compiles and preserves data in the overlapping region after resize.
 
### Related issues / PRs


<!-- Link any related issues or PRs. -->

### Changelog Entry
<!--
If this PR would require a changelog entry, add one here, or choose one of the following to add:
  - Not required
  - Unsure
-->

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
