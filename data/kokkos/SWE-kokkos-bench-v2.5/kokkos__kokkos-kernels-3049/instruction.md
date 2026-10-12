Fix the following issue in the Kokkos repository.

Add the batched scalar Givens rotation interface `KokkosBatched::Rotg::invoke(const SViewType& a, const SViewType& b, const MViewType& c, const SViewType& s)` returning `int`. All arguments must be rank-0 non-const views: `a` and `b` share a real or complex scalar type (`SViewType`); `c` is a real scalar magnitude (`MViewType`); `s` matches `SViewType`. The rotation must satisfy `[[c, s], [-conj(s), c]] * [[a], [b]] = [[r], [0]]`, with `a` overwritten by `r`. For real overloads, `b` must be updated by the result; for complex overloads, `b` must remain unchanged. Only the scalar `Rotg` interface is required; do not provide `SerialRotg` or `TeamRotg`. Refactor the internal BLAS `rotg_impl` to eliminate overflow for both real and complex scalar pointers, safely handling cases where `|a|` or `|b|` is zero or extremely small through appropriate scaling and norm computation, and preserve the real/complex distinction in `b` updates.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
