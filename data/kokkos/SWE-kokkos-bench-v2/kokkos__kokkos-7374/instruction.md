Fix the following issue in the Kokkos repository.

Working towards #7037 
Affects error reporting on invalid number of begin/end arguments on unmanaged OffsetView constructor and on out of bounds accesses.
The regexes are painful but they did catch a misprint in the error.  Let me know if you want to simplify or even ignore the messages.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
