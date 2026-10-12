Fix the following issue in the Kokkos repository.

Some accessor types (e.g., Sacado FAD with non-consecutive, span-strided elements) need mapping information—particularly the span size—to construct correctly from an AccessorArg_t. Introduce an ADL-discoverable customization point in the accessor's namespace that accepts a mapping and an AccessorArg_t and yields the accessor. The default overload must build the accessor from the argument value alone, without requiring mapping data. Update BasicView constructors that initialize an accessor from AccessorArg_t (both with and without an accompanying pointer argument) to invoke this customization point and pass the view's mapping. For consistency, also make the existing allocation_size_from_mapping_and_accessor customization point constexpr. The change should preserve compatibility for accessors that do not need mapping-based construction.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
