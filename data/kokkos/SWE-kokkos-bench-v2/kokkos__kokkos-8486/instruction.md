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
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
