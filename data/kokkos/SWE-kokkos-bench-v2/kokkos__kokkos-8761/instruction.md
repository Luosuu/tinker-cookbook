Fix the following issue in the Kokkos repository.

Split-up from #8689.
`ScatterValue` wraps a reference and the current implementation only implements a move constructor out of all the special member functions. This move constructor behaves like a copy constructor. Thus, the moved from object still impacts the moved-to object which appears unintended. Instead, this pull request proposes to delete the copy constructor and copy assignment operator implying that the move constructor and move assignment operator are also implicitly deleted.

We do some more cleanup by removing the (unused) `join` operator as was already mentioned in a comment.

Drive-by:  Remove `ScatterAccess`'s default destructor.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
