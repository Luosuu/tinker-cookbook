Fix the following issue in the Kokkos repository.

Add the missing Kokkos binary comparison predicates under classification and comparison: `isgreater`, `isgreaterequal`, `isless`, `islessequal`, `islessgreater`, and `isunordered`. Each takes two arithmetic arguments and returns `bool`. Provide overloads for `float`, `double`, `long double`, and a generic arithmetic-template overload that promotes mixed types and returns `bool`. Semantics must match the C math library: `isgreater` (`x > y`), `isgreaterequal` (`x >= y`), `isless` (`x < y`), `islessequal` (`x <= y`), `islessgreater` (`x < y || x > y`, false for equality or unordered pairs), and `isunordered` (true if either argument is NaN). The predicates must correctly handle NaN, infinity, and unordered comparisons according to standard rules, distinguishing them from relational operators.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
