Fix the following issue in the Kokkos repository.

In the public concepts header, export C++20 concepts named ExecutionSpace, MemorySpace, ExecutionPolicy, Reducer, and TeamHandle that correspond exactly to the existing traits is_execution_space, is_memory_space, is_execution_policy, is_reducer, and is_team_handle. Do not create concepts for other traits such as array_layout, memory_traits, work_item_property, hooks_policy, thread_team_member, host_thread_team_member, or graph_kernel; those remain type traits only. Export these five concept names in the Kokkos core module interface. In the parallel headers (e.g., Kokkos_Parallel.hpp and Kokkos_Parallel_Reduce.hpp), replace std::enable_if_t and other SFINAE conditions based on is_execution_policy or is_reducer with direct template constraints and conditional overloads using the ExecutionPolicy and Reducer concepts. The concepts must preserve the traits’ reference and pointer semantics: they must be false for reference and pointer types (e.g., ExecutionSpace<T&> is false) and true for qualifying value types, matching the existing trait behavior. No other production behavior changes are required.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
