Fix the following issue in the Kokkos repository.

Feature Request: Add loop unroll semantics to Range policy loops

It would be nice to have semantics for compile time loop unrolling for Range policy. This especially becomes important on memory bound loops where instruction level parallelism in functors is very low. See example below (copied from [here](https://github.com/kokkos/kokkos/issues/8033#issuecomment-2837137148))

 [implementation suggestion omitted]

I get the following benchmarks on A100 
[implementation suggestion omitted]

The loop unrolling parameter could be passed at compile tile to range policy. Similar to `Kokkos::ChunkSize`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
