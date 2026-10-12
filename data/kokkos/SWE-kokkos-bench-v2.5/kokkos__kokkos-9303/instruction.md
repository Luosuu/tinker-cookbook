Fix the following issue in the Kokkos repository.

Add or extend Kokkos::subview so that Views built with mdspan-style explicit template arguments (element type, extents, layout, accessor) derive their result through submdspan on the original view’s underlying mdspan using the provided slice arguments. The overload must preserve the original element type, index_type, and accessor policy, and the returned View’s mdspan_type must match the submdspan result. Required behavior: (1) support all standard slice forms (ALL_t, integer index, pair range); (2) for rank < 2, padded layouts (layout_left_padded / layout_right_padded) must simplify to layout_left / layout_right; (3) for rank >= 2, when slicing includes a pair-range on the final dimension, padded layouts must be retained rather than falling back to layout_stride; (4) classic-style View subview semantics must remain completely unaffected. The change applies only to mdspan-style View constructions and must not alter legacy behavior.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
