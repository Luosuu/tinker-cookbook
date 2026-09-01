Fix the following issue in the Kokkos repository.

Add `Kokkos_TypeInfo.hpp` defining `Kokkos::Impl::TypeInfo<T>` with a static `name()` member returning `std::string_view`, declared `constexpr` and `noexcept` on supported platforms. Introduce `KOKKOS_ENABLE_IMPL_TYPEINFO`, enabled only for verified toolchains: supported GCC/Clang/MSVC, Intel Classic with build date >= 20210228, NVCC >= 11.3.0; explicitly disabled for NVCC with an MSVC host and all older or unverified versions. When disabled, `name()` returns `"not supported"`. When enabled, the result must resolve type aliases, suppress anonymous-namespace prefixes that vary by compiler, preserve Clang lambda source-location annotations, and never begin with `"const "`; callers must apply `std::remove_const_t` before passing types. Wherever the library constructs kernel or execution-policy labels from functor or tag types, prefer `TypeInfo<std::remove_const_t<...>>::name()` over `typeid(...).name()` when the macro is defined, falling back to existing behavior otherwise. The output must remain valid as a kernel label.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
