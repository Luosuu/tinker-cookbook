Fix the following issue in the Kokkos repository.

Add KokkosSparse::Experimental::SellMatrix, a sparse format for matrices with balanced nonzeros per row that enables faster matrix-vector operations. Provide the class template parameterized by scalar type, signed ordinal type, device, optional memory traits, and size type inside namespace KokkosSparse::Experimental.

Expose nested aliases: execution_space, memory_space, device_type, memory_traits; value_type, ordinal_type, size_type; offsets_type, entries_type, and values_type as layout-right device views with the specified traits; const and non-const view variants; plus host_mirror_type and const_type.

Public members must include row and column counts, rows per slice, slice count, nnz, sell_nnz (padded nnz), and the corresponding view members slice_offsets, entries, and values.

The ordinal type must be signed. Provide a default constructor and a parameterized constructor accepting dimensions, nnz, padded nnz, rows per slice, and the three views. Enforce single-slice usage: rows_per_slice must be at least the number of rows. Reject configurations where slice offsets extent does not equal slice count plus one, where padded nnz is smaller than nnz, or where entries and values extents do not match padded nnz.

Also provide is_sell_matrix and is_sell_matrix_v traits in the same namespace to identify this format.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
