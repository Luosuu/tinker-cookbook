Fix the following issue in the Kokkos repository.

We currently toggle refcounting before and after copying functors in `parallel_*` operations. This isn't exception safe and a functor that throws in its copy constructor can break refcounting until the next `parallel_*` operation. This PR fixes the issue and provides tests.

Note this bug affects all backends.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
