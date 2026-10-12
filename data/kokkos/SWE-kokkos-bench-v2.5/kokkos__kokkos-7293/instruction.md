Fix the following issue in the Kokkos repository.

In the Kokkos namespace, add constexpr, noexcept free functions begin() and end() for Kokkos::Array<T, N>. Provide non-const overloads returning T* and const overloads returning const T*; all must carry __host__ __device__ annotations (e.g., KOKKOS_FUNCTION). They must work for every N, including N==0 where they return nullptr, and must be usable in constexpr contexts and with range-based for loops. To enable constexpr usage, both overloads of Array<T, 0>::data() must also become constexpr. Member begin/end may be added optionally, but the annotated free functions are required because std::begin/std::end lack the necessary execution-space annotations.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
