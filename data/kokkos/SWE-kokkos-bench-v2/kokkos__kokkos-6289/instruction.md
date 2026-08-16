Fix the following issue in the Kokkos repository.

We currently toggle refcounting before and after copying functors in `parallel_*` operations. This isn't exception safe and a functor that throws in its copy constructor can break refcounting until the next `parallel_*` operation. This PR fixes the issue and provides tests.

Note this bug affects all backends.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
