Fix the following issue in the Kokkos repository.

This PR allows users to interact with the underlying backend graph node.

It is an interoperability feature, hence the `<backend>_node` naming.

For CUDA and HIP, I could easily demonstrate a simple yet meaningful use case: enabling/disabling a node in-between submission.

~~For SYCL, I haven't experienced with it yet. So I'd suggest to go forward with this PR, we can deal with the SYCL case later.~~ SYCL is supported too.

### Changelog Entry

Definitely a changelog entry.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
