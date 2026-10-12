Fix the following issue in the Kokkos repository.

In Kokkos::Experimental::Graph, rename the underlying vendor-graph accessors from native_graph / native_graph_exec to backend-specific names and remove the old names. For Kokkos::Cuda provide cuda_graph() and cuda_graph_exec(); for Kokkos::HIP provide hip_graph() and hip_graph_exec(); for Kokkos::SYCL provide sycl_graph() and sycl_graph_exec(). These members must be const-qualified. They must expose the native graph and executable-graph handles without allowing mutation of the internal state held by Graph. For CUDA and HIP return the native pointer handles by value. For SYCL return the underlying command-graph objects by const reference (the graph object and its optional executable counterpart). This is a breaking rename with no required backward-compatibility aliases.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
