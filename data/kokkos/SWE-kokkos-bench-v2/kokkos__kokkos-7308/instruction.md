Fix the following issue in the Kokkos repository.

layout() function broken for Views with padding -- path forward

We have an issue because the LayoutLeft and LayoutRight class itself (not the mapping) has no place to store stride. As a consequence the following code is wrong:

[implementation suggestion omitted]

Suggested path forward:

- restrict `layout()` to `Kokkos::LayoutLeft/Right/Stride` and deprecate
  - error out when calling `layout()` where `stride!=extent`
- introduce `View::mapping()` -> return mdspan mappings
- introduce `View(data_handle, mdspan_mapping_type map)`

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
