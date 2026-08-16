Fix the following issue in the Kokkos repository.

- fix offset type of SpaceAwareAccesor
- Improved test to catch the mistake, and added a typedef for nested_accessor which I believe every wrapping accessor should have
- fix convertibility properties to match View convertibility

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
