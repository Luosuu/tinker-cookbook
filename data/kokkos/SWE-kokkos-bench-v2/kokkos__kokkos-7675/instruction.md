Fix the following issue in the Kokkos repository.

The status quo is a mixed bag of not reporting the violation, throwing a runtime error, etc.
I propose that we abort in all cases, regardless of the range being empty.
This PR is limited to range policies.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
