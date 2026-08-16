Fix the following issue in the Kokkos repository.

@dalg24 rightly pointed out in 
* https://github.com/kokkos/kokkos/pull/6904#discussion_r1543196102

> I am tempted to suggest to call these `get_cuda_graph[_exec]` because I am not convinced there is much generic code one could write that get the native graph nor that the concept of "executable" graph will always make sense.

> That's my point, it is more about interoperability than portability here.

At the time of #6904, we followed @masterleinad suggestion https://github.com/kokkos/kokkos/pull/6904#discussion_r1763256123 and converged to `native_graph[_exec]`, *i.e.* same API name for all backends.

Together with @maartenarnst, we acknowledge that naming the "get underlying vendor graph[exec]" as `native_graph[_exec]` was a wrong choice. The getters are about interoperability, not portability.

This PR renames the accessors as `<backend>_graph[_exec]`, to follow `Kokkos` naming convention for interoperability features.

Beyond renaming:
1. The accessors are now `const`-qualified functions, much like `Kokkos::Cuda::cuda_stream` for instance.
2. The returned object type does not allow the user to change the handles stored in `Kokkos::Experimental::Graph`, to ensure they don't break any invariant.

### Related issues / PRs

* https://github.com/kokkos/kokkos/pull/6904

### Changelog Entry

Not sure anyone uses these already, but probably worth a changelog entry nevertheless.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
