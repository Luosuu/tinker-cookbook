Fix the following issue in the Kokkos repository.

`ElementType` was a redundant template parameter. We should get it from `NestedAccessor`.

See https://eel.is/c++draft/linalg.conj.conjugatedaccessor for an example from the standard.

Motivation: streamline View internals somewhat.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
