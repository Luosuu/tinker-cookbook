Fix the following issue in the Kokkos repository.

In Sacado the accessor construction may require information from the mapping (specifically the span size) in some cases. Specifically it needs it if the elements of a FAD type are not consecutive but strided by the span size.

The customization point works like the other ones via ADL finding an overload in the same namespace as the accessor.
The default implementation constructs the accessor from `AccessorArg_t` value. 

[implementation suggestion omitted]

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
