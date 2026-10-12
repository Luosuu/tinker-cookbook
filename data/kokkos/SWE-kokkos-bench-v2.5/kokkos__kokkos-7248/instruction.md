Fix the following issue in the Kokkos repository.

Add overloads of `Kokkos::Experimental::create_graph` that do not take a closure. Provide a template overload that accepts an optional execution-space instance and defaults to `Kokkos::DefaultExecutionSpace`, returning `Kokkos::Graph<ExecutionSpace>`. It must be callable as `create_graph()` (using the default space) or `create_graph(exec)` (using the provided space). Constrain the existing closure-based overload so execution-space types are rejected as closures, preventing ambiguity. The returned empty graph must be usable for manual node construction and submission within the associated execution space.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
