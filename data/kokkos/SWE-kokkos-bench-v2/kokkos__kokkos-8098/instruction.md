Fix the following issue in the Kokkos repository.

This uses ADL based function lookup using functions that take a Kokkos::Impl::ViewArguments (effectively typelist) arg.

Thus its not yet fully public, but can easily be made public. It returns another special argument that speifies index_type and accessor_type.

This first PR only enables the customization of the accessor type effectively. It does not have the capability to pass arguments through to that accessor during construction.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
