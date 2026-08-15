Fix the following issue in the Kokkos repository.

We discussed in the developer meeting that we want to make `Kokkos_ScopeGuard.hpp` and `Kokkos_InitializeFinalize.hpp` public since these headers are very small and users might have translation units (like those containing `main`) that don't need more functionality than provided in these headers.
~~Drive-by: Rename `Kokkos_Core.cpp` to `Kokkos_InitializeFinalize.cpp`.~~

### Changelog Entry
 - Yes, needs one.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
