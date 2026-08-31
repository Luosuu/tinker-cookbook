Fix the following issue in the Kokkos repository.

This PR aims at adding Team and TeamVector implementations of batched Iamax.

- [x] Adding `Team` and `TeamVector` implementations of batched Iamax
- [x] Add unit-tests for them
- [x] Integrate `Serial`, `Team` and `TeamVector` implementations in a single file

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
