Fix the following issue in the Kokkos repository.

Current status of mathematical functions

Summarize below what is currently provided by Kokkos and what is missing.

x means already there
X.Y denotes functions that have been added more recently in version X.Y

# Basic operations
functions | Kokkos
--------- | ------
`abs` | x
`fabs` | x
`fmod` | x
`remainder` | x
`remquo` | 5.1
`fma` | x
`fmax` | x
`fmin` | x
`fdmin` | x
`nan` | x

# Exponential functions
functions | Kokkos
--------- | ------
`exp` | x
`exp2` | x
`expm1` | x
`log` | x
`log10` | x
`log2` | x
`log1p` | x

# Power functions
functions | Kokkos
--------- | ------
`pow` | x
`sqrt` | x
`cbrt` | x
`hypot` | x

# Trigonometric functions
functions | Kokkos
--------- | ------
`sin` | x
`cos` | x
`tan` | x
`asin` | x
`acos` | x
`atan` | x
`atan2` | x
# Hyperbolic functions
functions | Kokkos
--------- | ------
`sinh` | x
`cosh` | x
`tanh` | x
`asinh` | x
`acosh` | x
`atanh` | x

# Error and gamma functions
functions | Kokkos
--------- | ------
`erf` | x
`erfc` | x
`tgamma` | x
`lgamma` | x

# Nearest integer floating point operations
functions | Kokkos | notes
--------- | ------ | -----
`ceil` | x
`floor` | x
`trunc` | x
`round` | x
`lround` | 5.1 | `round(Arithmetic arg) -> long` not available in SYCL
`llround` | 5.1 | `round(Arithmetic arg) -> long long` not available in SYCL
`nearbyint` | x | not available in SYCL
`rint` | 5.1 | `rint(Arithmetic arg) -> Promoted`
`lrint` | 5.1 | `rint(Arithmetic arg) -> long long` not available in SYCL
`llrint` | 5.1 | `rint(Arithmetic arg) -> long` not available in SYCL

# Floating point manipulation functions
functions | Kokkos | notes
--------- | ------ | -----
`frexp` | | `frexp(Arithmetic arg, int* exp) -> Promoted` available in CUDA and SYCL
`ldexp` | | `ldexp(Arithmetic arg, int exp) -> Promoted` available in CUDA and SYCL
`modf` | 5.1 | `modf(FloatingPoint arg, FloatingPoint* iptr) -> FloatingPoint` available in CUDA and SYCL
`scalbn` | | `scalbn(Arithmetic arg, int exp) -> Promoted` available in CUDA
`scalbln` | | `scalbln(Arithmetic arg, long exp) -> Promoted` available in CUDA
`ilogb` | 5.1
`logb` | x
`nextafter` | x
`nexttoward` | | `nexttoward(Arithmetic1 from, Arithmetic2 to) -> Promoted` missing in CUDA and SYCL
`copysign` | x

# Classification and comparison
functions | Kokkos
--------- | ------
`fpclassify` | 5.1
`isfinite` | x
`isinf` | x
`isnan` | x
`isnormal` | 5.1
`signbit` | x
`isgreater` | 5.1
`isgreaterequal` | 5.1
`isless` | 5.1
`islessequal` | 5.1
`islessgreater` | 5.1
`isunordered` | 5.1

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
