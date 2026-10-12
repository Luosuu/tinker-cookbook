Fix the following issue in the Kokkos repository.

Fix ambiguous overload resolution in Kokkos::resize when called with an explicit ExecutionSpace instance and a layout argument (e.g., array_layout). The redundant overload taking an explicit execution-space reference, a view reference, and a layout conflicts with the generic property-based overload; remove it. The remaining generic overload must correctly interpret execution-space arguments through view-property construction and apply the requested layout. After removal, resize with an execution space and layout must compile without ambiguity, resize the view to the specified layout, and preserve existing data in the overlapping region. Maintain backward compatibility with all existing property-based resize usage.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
