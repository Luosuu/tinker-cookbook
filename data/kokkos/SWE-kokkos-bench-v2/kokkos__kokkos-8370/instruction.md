Fix the following issue in the Kokkos repository.

The problem here was an invalid handling of the stride member in the layout struct (not the mapping) for LayoutRight when converting View to DynRankView in particular. This did not happen for everything (e.g. rank-3 was fine, but rank-2 failed ...).

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
