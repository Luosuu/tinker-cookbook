Fix the following issue in the Kokkos repository.

Prevent segfaults in PyKokkos by enforcing memory contiguity checks on all public view-creation paths: pykokkos.array, from_numpy, from_array, and kernel arguments that accept array-like inputs (e.g., view= in parallel_for). Inputs must be either C-contiguous or F-contiguous. For arrays with ndim > 1 and no explicit layout argument, infer LayoutRight for C-contiguous and LayoutLeft for F-contiguous. Any array that is neither C- nor F-contiguous must raise ValueError. In array(), if the input is not a recognized np.ndarray, scalar, or array object, attempt conversion via np.asarray(); apply the contiguity check afterward, and raise TypeError if the type remains unsupported. This ensures kernel calls safely reject non-contiguous numpy arrays instead of causing a segfault.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
