Fix the following issue in the Kokkos repository.

Combined reducer fail on Cuda backends with `int16_t` and `int8_t`

When calling a combined reducer with `int16_t` or `int8_t` values for the reducer, the CUDA backend always returns 0 for certain amounts of arguments.

For instance, the following code:
[implementation suggestion omitted]

Produces:
[implementation suggestion omitted]

It looks like the combined reducer can't return a value when the size of the combined type is more than 32 bits but not a multiple of 32.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
