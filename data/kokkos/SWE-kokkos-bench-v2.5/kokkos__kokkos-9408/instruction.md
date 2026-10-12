Fix the following issue in the Kokkos repository.

Remove the redundant `ElementType` template parameter from `ReferenceCountedAccessor`, its trait specializations, associated data-handle friend declarations, and the `CheckedReferenceCountedAccessor` and `CheckedReferenceCountedRelaxedAtomicAccessor` aliases. Derive `element_type`, `data_handle_type`, `reference`, and `offset_policy` from the `NestedAccessor` argument. Update conversion constructors to compare `NestedAccessor::element_type` instead of the removed explicit parameter. This streamlines View internals by adopting the nested-accessor pattern where the wrapped accessor supplies its own element type.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
