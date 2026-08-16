Fix the following issue in the Kokkos repository.

Working towards #7037 
Affects error reporting on invalid number of begin/end arguments on unmanaged OffsetView constructor and on out of bounds accesses.
The regexes are painful but they did catch a misprint in the error.  Let me know if you want to simplify or even ignore the messages.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
