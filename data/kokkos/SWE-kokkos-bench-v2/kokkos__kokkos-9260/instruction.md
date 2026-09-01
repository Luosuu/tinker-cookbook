Fix the following issue in the Kokkos repository.

Add Kokkos::Experimental::BadAlloc, derived from std::runtime_error, and export it in the Kokkos::Experimental namespace. Provide a constructor that takes the memory-space name, allocation size, and label, with corresponding const accessors. Its what() must include the memory-space name, a human-readable representation of the size, and the label. Update the library’s bad-allocation throw path to use BadAlloc instead of std::runtime_error so existing catch (std::runtime_error&) blocks remain compatible.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
