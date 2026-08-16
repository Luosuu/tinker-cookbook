Fix the following issue in the Kokkos repository.

## Summary

This PR adds a `then` node to `Kokkos::Graph`.

## Description

The idea is that this `then` node mimics the `then` in [P2300](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-sender-adaptor-then).

In essence, you pass a functor to the `then` and it's guaranteed to be executed at each graph submission, in the graph topology ordering.

[implementation suggestion omitted]

Things to be noted from the above code snippet:
1. The functor passed to the `then` must be callable without any argument, marked with `KOKKOS_FUNCTION`.
2. The `then` is nothing else then a kernel launch, done through one of `Kokkos` drivers.
3. It adds exactly one node in the `Kokkos` graph and one node in the backend graph.

:warning: For now, I've implemented it as a parallel-for from 0 to 1, but I think we can do better than that (in the future).

## Related

- extracted from #7552

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
