Fix the following issue in the Kokkos repository.

Add an ADL-based customization point in Kokkos::Impl for mdspan-backed Views. Provide Kokkos::Impl::ViewArguments and Kokkos::Impl::ViewCustomArguments<IndexType, AccessorType>. The default Kokkos::Impl::customize_view_arguments accepts ViewArguments and returns void. Users may provide ADL-discoverable overloads that accept ViewArguments and return ViewCustomArguments to specify a custom index_type and accessor_type. Introduce Kokkos::ViewTraits::impl_is_customized as a constexpr bool that is true when customization is active for the value/array/device/memory combination and false otherwise. When active, the view's extents must use the specified index_type and the specified accessor_type; otherwise use defaults. This feature must be confined to the mdspan-enabled implementation path and must not change legacy View behavior. Only accessor customization is included; passing arguments through to the accessor during construction is not supported.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
