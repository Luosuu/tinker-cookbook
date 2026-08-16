Fix the following issue in the Kokkos repository.

This PR aims at adding Team and TeamVector implementations of batched Iamax.

- [x] Adding `Team` and `TeamVector` implementations of batched Iamax
- [x] Add unit-tests for them
- [x] Integrate `Serial`, `Team` and `TeamVector` implementations in a single file

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
