Fix the following issue in the Kokkos repository.

Add mdspan-compatible member typedefs to the legacy View class: element_type, index_type, rank_type, data_handle_type, and reference, mapped from the corresponding existing value, memory-space size, rank, pointer, and reference types. Fix const propagation in all uniform const-view typedefs—uniform_const_type, uniform_runtime_const_type, uniform_const_nomemspace_type, and uniform_runtime_const_nomemspace_type—so that const applies to the element being accessed (yielding a data/access pattern equivalent to const T*) instead of making the pointer const (T* const). All four variants, including the nomemspace forms, must behave consistently, and existing non-const uniform typedefs must remain unchanged.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
