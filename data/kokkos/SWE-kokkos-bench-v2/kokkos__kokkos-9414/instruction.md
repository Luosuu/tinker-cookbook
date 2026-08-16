Fix the following issue in the Kokkos repository.

`CheckedRelaxedAtomicAccessor` ignores `MemoryScope` parameter

[source location omitted]

`AtomicAccessorRelaxed` has a MemoryScope argument:

[source location omitted]

I think we just need to use it:

[implementation suggestion omitted]

(same for `CheckedReferenceCountedRelaxedAtomicAccessor`)

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
