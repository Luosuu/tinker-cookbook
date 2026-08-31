Fix the following issue in the Kokkos repository.

This PR aims at improving the implementation and adding unit-tests of Team/TeamVector `copy`. See also #2910

 - [x] Adding unit-tests for `TeamCopy` and `TeamVectorCopy`
 - [x] Supporting `ConjTrans` operator
 - [x] Fixing the `TeamCopy` and `TeamVectorCopy` `impl` for Transpose case

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
