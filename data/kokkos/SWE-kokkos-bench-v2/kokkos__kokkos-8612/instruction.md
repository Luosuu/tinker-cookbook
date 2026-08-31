Fix the following issue in the Kokkos repository.

This fixes a bug that @crtrott pointed out:

[implementation suggestion omitted]

The issue is here:

[implementation suggestion omitted]

Where data_handle satisfies the constraints as it is copy-constructible, yet gets cast to `pointer_type` in the initialization list and so loses its reference count.

This PR adds a check for _implicit_ convertibility to `pointer_type` which is imo the correct constraint here.

Note that after this PR `Kokkos::View<int*> b(a.data_handle(), 5);` won't compile because there is no suitable constructor. I can work on a followup PR to add it (legacy view did not have such a constructor anyway).

I would like this to be squeezed into 5.0 if possible.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
