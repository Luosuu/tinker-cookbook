Fix the following issue in the Kokkos repository.

The goal of this PR is to ensure that in the default implementation of the `Kokkos::Graph`, fencing occurs as needed to ensure that dependencies are met when using an aggregate node.

The default implementation fences when the predecessor is awaitable and not on the same execution space instance. Currently, the default implementation considers that an aggregate node is *not* awaitable. However, if an aggregate doesn't fence because of equality of execution space instances, a child node of the aggregate actually may have to fence if it is on a different execution space instance. And so it appears that aggregate nodes must in fact be considered "awaitable".

This PR thus modifies the `awaitable` function so that aggregate nodes are awaitable. The new test illustrates a case in which a child node of an aggregate node needs to fence.

EDIT: Motivated by a failing test seen in our downstream code on HPX. 

Joint work with @romintomasetti.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
