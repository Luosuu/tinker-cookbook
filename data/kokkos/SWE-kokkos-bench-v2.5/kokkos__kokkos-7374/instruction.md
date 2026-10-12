Fix the following issue in the Kokkos repository.

For Kokkos::Experimental::OffsetView, change precondition-failure handling from throwing runtime exceptions to terminating via Kokkos::abort. Apply this to the unmanaged constructor validation and to out-of-bounds element-access checks.

Unmanaged constructor: verify that the begin/end argument lists match the view rank; that each end - begin is non-negative; and that the subtraction does not overflow. On failure, call Kokkos::abort with a message that starts with 'Kokkos::Experimental::OffsetView ERROR: for unmanaged OffsetView', reports the actual list sizes correctly (fixing the misprint that reports ends.size() as begins.size()), and retains the required descriptive phrases (e.g., 'must be non-negative', 'overflows'). Do not simplify or omit these messages.

Bounds checks: on both host and device access paths, replace any throw with Kokkos::abort and preserve the observable diagnostic content, including failure keywords.

This changes only error-handling behavior; valid construction and access remain unchanged.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
