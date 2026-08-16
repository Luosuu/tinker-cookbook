Fix the following issue in the Kokkos repository.

Implementing the changes discussed in #7449
There was no objections at the developer meeting to proceed with all the deprecations that were proposed.
Fell free to speak up if you disagree with any of it.

Summary:
* Deprecated `atomic_query_version()` (no known usage and unclear how that is useful)
* Deprecated `atomic_assign()` in favor of `atomic_store()`
* Deprecated `atomic_compare_exchange_strong()`
* Deprecated `atomic_{increment, decrement}`
* ~~Deprecated `atomic_{fetch_nand, nand_fetch, nand}`~~ **withdrawn**
(I only feel strongly about the three first deprecation items)
* Added `atomic_{mod,xor,nand,lshift,rshift}` that were missing

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
