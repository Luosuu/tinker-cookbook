Fix the following issue in the Kokkos repository.

- fix offset type of SpaceAwareAccesor
- Improved test to catch the mistake, and added a typedef for nested_accessor which I believe every wrapping accessor should have
- fix convertibility properties to match View convertibility

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
