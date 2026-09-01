Fix the following issue in the Kokkos repository.

Enable scratch-space construction for customized View types. For views whose customization traits specify custom mapping or accessor behavior, provide scratch-memory constructors that take an execution_space scratch_memory_space followed by integral dimensions. One overload accepts exactly rank sizes: build the mapping from the sizes, construct the accessor through the customization traits, determine the required scratch allocation via the customization's size-computation rules, and initialize the base view from properly aligned scratch memory of that size. Another overload accepts exactly rank+1 sizes: form the mapping from the first rank sizes, and use the extra size as an accessor argument (e.g., ensemble dimension) via the customization's accessor-argument rules; then compute allocation and initialize the base view as above. The existing overload accepting up to eight size_t arguments (with defaults) must be limited to non-customized views, using the mapping's required span size for allocation and a default-constructed accessor. Constructors that take explicit layout, mapping, or accessor objects are excluded. Non-customized behavior must not change.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
