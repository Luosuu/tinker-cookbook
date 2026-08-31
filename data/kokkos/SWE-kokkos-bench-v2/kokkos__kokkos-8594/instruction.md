Fix the following issue in the Kokkos repository.

Lvalue requirement for Kokkos::Experimental::contribute

**Describe the bug**

Please provide a concise, clear description of the bug, as well as any available error logs.  Feel free to contact the Kokkos Slack `# build` channel for further discussion of your issue.

I think the requirement that `Kokkos::Experimental::contribute` accept only an lvalue for dest is superfluous. It doesn't permit using contribute with cases like e.g. an inline subview:

[implementation suggestion omitted]

Further in the implementation the function `contribute_into` accepts `const& dest` so this doesn't require any more changes in the code.

TL;DR  change Kokkos_ScatterView.hpp
[implementation suggestion omitted]
to
[implementation suggestion omitted]

**Please include the following for a minimal reproducer**

1. Compilers (with versions)
2. Kokkos release or commit used (i.e., the sha1 number)
3. Platform, architecture and backend
4. CMake configure command
5. Output from CMake configure command
6. Minimum, complete code needed to reproduce the bug
7. Command line needed to reproduce the bug
8. `KokkosCore_config.h` header file (generated during the build)
9. Please provide any additional relevant error logs

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
