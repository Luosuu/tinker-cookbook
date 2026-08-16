Fix the following issue in the Kokkos repository.

Cannot use SequentialHostInit with UnorderedMap

**Describe the bug**

Attempting to do:
[implementation suggestion omitted]
leads to compile errors because the `UnorderedMap` constructor appends `WithoutInitializing` to the property list passed to the constructor which hits a static assertion about not including both `WithoutInitializing` and `SequentialHostInit` in the same constructor properties.

**Please include the following for a minimal reproducer**
4.6.01 and develop both have the issue, can be reproduced with any compiler.

@crtrott @dalg24

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
