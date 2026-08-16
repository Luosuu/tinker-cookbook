Fix the following issue in the Kokkos repository.

This modernizes ErrorReporter and moves it out of Experimental.
- Make member functions follow naming conventions.
- Deprecate (under deprecated code 4) old names.
- Remove use of DualView.
- New get_reports() returns pair of std::vector.
- introduce default argument for device type

The biggest change is the `get_reports`

[implementation suggestion omitted]

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
