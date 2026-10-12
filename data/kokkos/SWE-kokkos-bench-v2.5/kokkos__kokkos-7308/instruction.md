Fix the following issue in the Kokkos repository.

Fix View::layout() for padded views: it must return the layout object carrying the view's dimensions and its actual stride, without erroring solely because stride differs from the default extent-based value. Add a public, mutable size_t stride member to LayoutLeft and LayoutRight, initialized to a default/unset value and directly assignable. View::layout() must populate this stride from the view. When constructing a View from one of these layout instances, use instance.stride if it is explicitly set; otherwise compute stride via the existing padding rules. Provide bidirectional mdspan interoperability: View::to_mdspan() must yield correct extents and strides, and View must be constructible from a data handle plus an mdspan mapping/object, reproducing extents and strides including for non-contiguous subviews. In layout-to-mdspan conversions, propagate stride for left/right padded layouts and preserve the stride array for LayoutStride. For LayoutRight converted to/from a right-padded mdspan layout, enforce that any custom stride equals the product of dimensions 1 through rank-1 when rank > 2; for rank 2 any custom stride is permitted. These layout classes remain deprecated in favor of mdspan mappings but must work correctly with explicit stride.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
