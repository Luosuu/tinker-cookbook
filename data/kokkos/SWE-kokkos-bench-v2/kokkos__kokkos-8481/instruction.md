Fix the following issue in the Kokkos repository.

Fix dependency fencing for aggregate nodes in the default `Kokkos::Graph` implementation. An aggregate node represents a `when_all` event and must be considered awaitable even though it launches no kernel itself. Only a root node is non-awaitable. When a successor runs on a different execution-space instance, the graph must therefore fence on the aggregate predecessor so every child dependency has completed; behavior for nodes on the same execution-space instance remains unchanged.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
