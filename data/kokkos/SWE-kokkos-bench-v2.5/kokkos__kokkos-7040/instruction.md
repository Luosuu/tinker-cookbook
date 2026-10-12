Fix the following issue in the Kokkos repository.

Enable structured binding support for Kokkos::complex<RealType> in host and device code without changing existing behavior. Provide std::tuple_size specialization yielding 2 and std::tuple_element<I, ...> specialization yielding RealType for I = 0 and I = 1, with a static_assert enforcing I < 2. In namespace Kokkos, provide constexpr, noexcept, KOKKOS_FUNCTION get<I> overloads for all reference categories of complex<RealType> (lvalue, const lvalue, rvalue, const rvalue), each enforcing I < 2 via static_assert. Index 0 refers to the real part and index 1 to the imaginary part, preserving the value category. All additions must work in constexpr and device contexts where supported and must not break existing Kokkos::complex usage.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
