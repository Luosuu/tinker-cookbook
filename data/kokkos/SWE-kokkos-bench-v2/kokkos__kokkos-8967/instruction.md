Fix the following issue in the Kokkos repository.

Create an unmanaged subview from a view is broken (rank-1)

Creating an unmanaged subview of a view as follows used to work until 8cfdfd8abe48b57cfb1b887fd8c9e3d77f990b51 from @crtrott :
[implementation suggestion omitted]

Reverting 8cfdfd8abe48b57cfb1b887fd8c9e3d77f990b51 makes the above code compile just fine. The question is, since there wasn't any test that captured the regression:
> Is it allowed to create an unmanaged view from a view as done above?

If so, this needs a fix @crtrott :wink: 

Compiler output:
[implementation suggestion omitted]

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
