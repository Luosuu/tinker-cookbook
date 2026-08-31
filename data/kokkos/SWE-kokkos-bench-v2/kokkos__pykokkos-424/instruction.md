Fix the following issue in the Kokkos repository.

## View Dtype Annotation Validation

`pykokkos/interface/parallel_dispatch.py` now validates explicit `View` dtype annotations against the actual dtype of the provided `View`.

### Changes

A new helper has been added to:

* Normalize PyKokkos and Python dtype aliases into comparable canonical names
* Extract the expected dtype from annotations such as `pk.View1D[pk.int32]`
* Compare the expected dtype against the actual `View.dtype`
* Raise a clear and informative `TypeError` when the dtypes do not match

### Examples

[implementation suggestion omitted]

Passing a `View` with dtype `int64` now raises a `TypeError` indicating the mismatch between the annotated and actual dtypes.

### Tests

Added `tests/test_view_dtype_mismatch.py` with coverage for:

#### Invalid dtype combinations

* Rejecting `int64` arrays passed to `pk.View1D[pk.int32]`
* Rejecting NumPy `dtype=int` arrays passed to `pk.View1D[int]`
* Rejecting `float64` arrays passed to `pk.View1D[pk.float]`
* Rejecting `float32` arrays passed to `pk.View1D[float]`

#### Valid dtype combinations

* Accepting `int32` arrays passed to `pk.View1D[pk.int32]`

#### Existing behavior preserved

* Preserving type inference for unannotated `int64` views

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
