Fix the following issue in the Kokkos repository.

Correct the alias templates `CheckedRelaxedAtomicAccessor` and `CheckedReferenceCountedRelaxedAtomicAccessor` so they honor the `MemoryScope` template parameter instead of ignoring it. Both aliases must forward their `MemoryScope` argument—defaulting to `desul::MemoryScopeDevice`—to the underlying `AtomicAccessorRelaxed<ElementType, MemoryScope>`. This ensures the requested memory scope is respected consistently for both relaxed atomic accessor aliases.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
