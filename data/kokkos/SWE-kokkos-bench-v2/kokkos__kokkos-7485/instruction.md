Fix the following issue in the Kokkos repository.

Matching the nature of value_type of Kokkos::RandomAccessIterator to that of C++ standard library

Acknowledging @masterleinad for figuring out the underlying issue and the fix.

Currently, the `value_type` of a `Kokkos::RandomAccessIterator` iterator is `const` if it is a const iterator like what is returned by `cbegin()`. However, the `value_type` of `std::vector.cbegin()` is non-`const`.

For example, on the develop branch with commit (81fc5622ad2d9b57427df91f2bee8b3a9e5c8ca9), the following `static_assert`s on the type of the `value_type`s will pass:

[implementation suggestion omitted]

Refer: https://godbolt.org/z/W64daPebd

Refer [here](https://cplusplus.github.io/LWG/issue322) and [here](https://stackoverflow.com/questions/12819405/why-is-stditerator-traitsvalue-type-non-const-even-for-a-const-iterator/12821204#12821204) for some motivation about why value_type is non-const in C++. In short, "constness doesn't matter for `value_type`, since a value implies a copy. The type of `std::iterator_traits<const T*>::reference` is `const T&` however".

It is better to make the `value_type` behave in the same manner as it is in the C++ standard. It will help with calling third party libraries when they are available (e.g., Nvidia Thrust). See below about how an error is encountered when working with `thrust::inclusive_scan`. This change will also help with more streamlined development of the Kokkos library.

In this issue, the type alias value_type will be changed from `typename view_type::value_type` to `typename view_type::non_const_value_type` in `class RandomAccessIterator` in `algorithms/src/std_algorithms/impl/Kokkos_RandomAccessIterator.hpp`.

According to the needs, existing tests will be modified and new tests will be added.

How to create error condition when working with current value_type of RandomAccessIterator?
Refer to https://github.com/science-enthusiast/kokkos/tree/thrust_inclusive_scan

In this branch, `algorithms/unit_tests/TestStdAlgorithmsInclusiveScan.cpp` and `algorithms/src/std_algorithms/Kokkos_InclusiveScan.hpp` are modified by replacing calls to `Kokkos::Experimental::cbegin()` and `Kokkos::Experimental::cend()` with `Kokkos::Experimental::begin()` and `Kokkos::Experimental::end()`.
The error condition can be recreated by modifying `run_single_scenario()` in `algorithms/unit_tests/TestStdAlgorithmsInclusiveScan.cpp` to pass `KE::cbegin()` and `KE::cend()` (which is how they are passed in the develop branch) instead of `KE::begin()` and `KE::end()`. The error log is [build_error.txt](https://github.com/user-attachments/files/17491221/build_error.txt).

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
