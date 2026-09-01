Fix the following issue in the Kokkos repository.

Fix and extend the public batched dense APIs `KokkosBatched::TeamCopy` and `KokkosBatched::TeamVectorCopy` (all `invoke(member, A, B)` specializations) to correctly support `NoTranspose`, `Transpose`, and `ConjTranspose` modes for both standard and strided batched views, without changing the public signatures.

Observed bugs:
- `Transpose` specializations apply incorrect input validation and do not properly exchange 2D extents or map strides (first input stride should map to second output stride and vice versa); they also mishandle 1D views and miss an immediate return for empty views.
- `NoTranspose` specializations misapply stride mapping for 2D views when the leading dimension equals 1.
- `ConjTranspose` specializations are missing entirely.

Required behavior:
- Add working `TeamCopy` and `TeamVectorCopy` specializations for `Trans::ConjTranspose`. They must validate conjugate-transpose inputs, return immediately for empty views, apply complex conjugation to copied values, and correctly swap extents and strides for 2D views, with proper 1D and unit-dimension handling.
- Repair the `Transpose` specializations with correct transpose validation, empty-view quick return, unchanged 1D behavior, and accurate 2D extent/stride exchange including cases where one extent is 1.
- Repair the `NoTranspose` specializations so that 2D edge cases with a unit leading dimension use the correct second-stride mapping instead of the first.
- The underlying copy execution must apply the appropriate linear operation—identity for `NoTranspose` and `Transpose`, conjugate for `ConjTranspose`.
- Preserve compatibility with existing deprecated wrapper structures.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
