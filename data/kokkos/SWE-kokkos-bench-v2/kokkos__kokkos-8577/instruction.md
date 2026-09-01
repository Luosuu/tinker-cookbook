Fix the following issue in the Kokkos repository.

Add STL-compatible iterator access to Kokkos::Array<T, N>, including its N==0 specialization. Introduce member types `pointer` (`T*`) and `const_pointer` (`T const*`). Provide `constexpr noexcept` overloads: non-const `begin()` and `end()` returning `pointer`; const `begin()` and `end()` returning `const_pointer`; and `cbegin()`/`cend()` returning `const_pointer`. For N==0 all four return `nullptr`. For N>0, `begin()` addresses the first element, `end()` addresses one past the last (`begin() + N`), and the const variants match. Ensure interoperability with C++ standard algorithms.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
