Fix the following issue in the Kokkos repository.

Implement the atomic API updates: deprecate `atomic_query_version()` without a replacement, `atomic_assign()` in favor of `atomic_store()`, `atomic_compare_exchange_strong()` in favor of `atomic_compare_exchange()`, `atomic_increment()` in favor of `atomic_inc()`, and `atomic_decrement()` in favor of `atomic_dec()`. Do not deprecate `atomic_fetch_nand`, `nand_fetch`, or `nand`. Guard deprecated APIs behind `KOKKOS_ENABLE_DEPRECATED_CODE_4` and annotate them with the repository's standard deprecation mechanism and replacement guidance. Add the missing void-returning atomic updates that discard the old value: `atomic_mod`, `atomic_xor`, `atomic_nand`, `atomic_lshift`, and `atomic_rshift`. Update remaining production-library call sites to use the replacement APIs.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
