Fix the following issue in the Kokkos repository.

Add backend-interoperability accessors to Kokkos::Experimental::GraphNodeRef and Kokkos::Experimental::Graph using the <backend>_node and <backend>_graph_exec naming: cuda_node, hip_node, sycl_node on node references, and cuda_graph_exec, hip_graph_exec, sycl_graph_exec on graphs. Availability requires the ExecutionSpace to match the backend and the corresponding backend enable/graph macro to be set (for SYCL, KOKKOS_ENABLE_SYCL with KOKKOS_IMPL_SYCL_GRAPH_SUPPORT). The node accessors expose native vendor graph node handles; the graph accessors expose native executable graph handles. For SYCL, the graph accessor returns an optional-like object with .has_value() and operator-> to the native command_graph; using its ->update() requires the graph was created with the updatable property, else it throws. This allows direct native manipulation—e.g., enabling or disabling nodes—between submissions. A changelog entry is required.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
