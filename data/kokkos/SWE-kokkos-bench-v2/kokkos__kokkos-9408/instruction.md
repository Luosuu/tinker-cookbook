Fix the following issue in the Kokkos repository.

`ElementType` was a redundant template parameter. We should get it from `NestedAccessor`.

See https://eel.is/c++draft/linalg.conj.conjugatedaccessor for an example from the standard.

Motivation: streamline View internals somewhat.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
