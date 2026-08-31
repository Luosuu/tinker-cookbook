Fix the following issue in the Kokkos repository.

This uses ADL based function lookup using functions that take a Kokkos::Impl::ViewArguments (effectively typelist) arg.

Thus its not yet fully public, but can easily be made public. It returns another special argument that speifies index_type and accessor_type.

This first PR only enables the customization of the accessor type effectively. It does not have the capability to pass arguments through to that accessor during construction.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
