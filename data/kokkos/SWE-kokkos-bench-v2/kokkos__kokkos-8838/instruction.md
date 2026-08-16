Fix the following issue in the Kokkos repository.

We had an abort in there for layout_right_padded to LayoutRight which only actually applies to LegacyView since they were different layouts. With the new View implementation they are the same.

A second bug was in DynRankView where something I thought only applies to Sacado, actually applies always. We just didn't test the case where we ran into trouble.

I added testing for all of this - with the test also demonstrating the change in behavior from Legacy to new view.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
