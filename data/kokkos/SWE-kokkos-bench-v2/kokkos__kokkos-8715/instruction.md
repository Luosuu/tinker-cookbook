Fix the following issue in the Kokkos repository.

As discussed in https://kokkosteam.slack.com/archives/G5CBLMFLP/p1764323555905049. Also adding tests for `BOr`, `LAnd` and `LOr` and testing with `unsigned int`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
