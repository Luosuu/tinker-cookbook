Fix the following issue in the Kokkos repository.

This PR adds an overload of `Kokkos::Experimental::create_graph` that does not take a closure.

This helps supporting advanced use cases for which creating the graph "in one shot" is not satisfactory.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
