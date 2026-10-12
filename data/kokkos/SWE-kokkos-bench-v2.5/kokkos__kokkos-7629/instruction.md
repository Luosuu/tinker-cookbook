Fix the following issue in the Kokkos repository.

Add `then` overloads to `Kokkos::Experimental::GraphNodeRef`: one taking `(Label, Functor)` and one taking `(Label, ExecutionSpace, Functor)`. A string label is required. The functor must return `void`, be callable with no arguments, and carry the `KOKKOS_FUNCTION` macro. On `graph.submit`, the node executes once in dependency order, adding exactly one node to both the public and backend graphs. Implement it as a driver launch over a single-element range (e.g., 0 to 1) so the functor receives no loop index. Internally, introduce a new graph-kernel node type recognized by `Kokkos::Impl::is_graph_kernel_v` and integrate it with the existing graph dependency-linking mechanism so it can be linked into the topology.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
