Fix the following issue in the Kokkos repository.

This PR aims at improving the implementation and adding unit-tests of Team/TeamVector `copy`. See also #2910

 - [x] Adding unit-tests for `TeamCopy` and `TeamVectorCopy`
 - [x] Supporting `ConjTrans` operator
 - [x] Fixing the `TeamCopy` and `TeamVectorCopy` `impl` for Transpose case

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
