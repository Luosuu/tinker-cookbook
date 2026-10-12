Fix the following issue in the Kokkos repository.

When KOKKOS_ENABLE_IMPL_MDSPAN is enabled, implement Kokkos::Impl::SpaceAwareAccessor<MemorySpace, NestedAccessor> with an AnonymousSpace specialization, and make it the default mdspan accessor for Kokkos::View via Impl::MDSpanViewTraits and View::to_mdspan(). This applies only to the mdspan layer, not to general View indexing. The wrapper must expose the nested accessor’s element_type, reference, and data_handle_type; define offset_policy as SpaceAwareAccessor carrying the nested offset policy; alias memory_space to MemorySpace (AnonymousSpace for the specialization); and be an empty type whenever the nested accessor permits it. For MemorySpace other than AnonymousSpace, access() must perform a runtime check that reports an error for an inaccessible memory space before delegating to the nested accessor; the AnonymousSpace specialization must skip this check entirely. Provide default construction, construction from NestedAccessor, and explicit conversion back to NestedAccessor. Support templated construction and conversion between SpaceAwareAccessor instances based on nested-accessor constructibility: AnonymousSpace instances must be compatible with any matching-space instance having a compatible nested accessor; same-space instances follow nested rules (e.g., const promotion allowed, incompatible element changes disallowed); cross-space conversion is allowed only under compatible nested-accessor and space-matching rules.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
