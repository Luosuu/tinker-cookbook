Fix the following issue in the Kokkos repository.

The status quo is a mixed bag of not reporting the violation, throwing a runtime error, etc.
I propose that we abort in all cases, regardless of the range being empty.
This PR is limited to range policies.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
