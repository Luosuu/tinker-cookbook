Fix the following issue in the Kokkos repository.

This PR introduces a signed `index_type` for each memory and execution space and standardizes `size_type` as unsigned across all Kokkos backends. This aligns Kokkos with the mdspan model where `index_type` is signed and `size_type` is unsigned.

### Motivation
- **Signed indexing**: Signed indices are safer for arithmetic (e.g., negative ranges) and avoid undefined behavior in overflow cases.
- **Consistency**: Different backends previously used different index/size semantics; this makes them uniform.
- **mdspan compatibility**: `std::mdspan` uses `index_type` (signed) and `size_type` (unsigned); Kokkos Views should mirror this.

### Changes
Memory spaces
- Add `index_type = std::make_signed_t<size_type>` to all memory spaces:
  - Cuda: `CudaSpace`, `CudaUVMSpace`, `CudaHostPinnedSpace`
  - HIP: `HIPSpace`, `HIPHostPinnedSpace`, `HIPManagedSpace`
  - Host: `HostSpace`, `AnonymousSpace`
  - OpenACC: `OpenACCSpace`
  - SYCL: `SYCLDeviceUSMSpace`, `SYCLSharedUSMSpace`, `SYCLHostUSMSpace`
Execution spaces
- Propagate `index_type` from memory space to execution spaces (Cuda, HIP, HPX, OpenACC, OpenMP, SYCL, Serial, Threads).
Policies
- Default policy `index_type` is now `execution_space::index_type` instead of `execution_space::size_type`.
- Improved overflow handling in `RangePolicy` for signed index types.
Views and CRS
- `BasicView`: Use `mdspan_type::index_type` for `index_type`.
- `ViewTraits`: Add `index_type`.
- `Crs`: Use `index_type` in transpose and count/fill utilities.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
