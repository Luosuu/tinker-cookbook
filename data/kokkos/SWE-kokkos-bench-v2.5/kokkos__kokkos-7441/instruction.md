Fix the following issue in the Kokkos repository.

Add a signed index_type (std::make_signed_t<size_type>) to every Kokkos memory space—including Cuda (all variants), HIP (all variants), Host, Anonymous, OpenACC, SYCL (all variants), ScratchMemorySpace, NextSiliconSharedSpace, and SYCLInternal—while keeping size_type unsigned everywhere. Propagate this signed index_type from the associated memory space to all execution spaces (Cuda, HIP, HPX, OpenACC, OpenMP, SYCL, Serial, Threads, NextSilicon, and any others).

Update RangePolicy so its default index type is the execution space’s index_type, not size_type. RangePolicy must support potentially negative work bounds. For each bound, enforce safe implicit conversions: a conversion is permitted only if it preserves the original value through round-trip casting; for arithmetic types with differing signs, the value must also fit within the target type’s limits. Unsafe conversions must abort with an error identifying the offending bound and the unsafe conversion.

Align Views with std::mdspan: BasicView should use mdspan_type::index_type (signed) for index_type and mdspan_type::size_type (unsigned) for size_type. Add index_type to ViewTraits, and update CRS transpose and count/fill utilities to use the signed index_type. Execution policy traits must derive index_type from execution_space::index_type.

For internal consistency, size-based allocation loops must explicitly use an unsigned RangePolicy (size_t index, zero-based bounds) rather than relying on signed defaults. In OpenACC parallel reduction, loops over signed index ranges must not carry sequential loop annotations.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
