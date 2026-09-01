Fix the following issue in the Kokkos repository.

Add a device-callable batched Rotmg interface for scalar real Givens rotation. Provide invoke accepting 0-D View scalars d1, d2, x1, y1 and a 1-D non-const real View param of length exactly 5; d1, d2, x1, and param are mutable, while y1 is input. The function updates d1, d2, x1, writes the five-element matrix (flag, h11, h21, h12, h22) into param, returns 0, supports only real types, and includes a debug check enforcing param length 5. Complex types are unsupported. Do not provide SerialRotmg or TeamRotmg. Refactor the BLAS rotmg_impl using mutable handles for d1, d2, x1, and param; correct the d1 < 0 path by setting flag to -1, zeroing d1, d2, and x1, then fully assigning all five param entries so the array is consistently populated regardless of branch.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
