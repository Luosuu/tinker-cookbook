Fix the following issue in the Kokkos repository.

Implementing DynRankView in a way that it is compatible with the current and the next impl of View gets a bit smoother by already introducing some of the mdspan typedefs (the ones which make trivially sense for now). In the process I implemented a test for all the View member typedefs and found that the const versions of the uniform typedefs are broken. So the last commit fixes that. 

Before `uniform_const_type` was effectively something like `View<T* const, ...>` instead of `View<const T*>` ...

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
