Fix the following issue in the Kokkos repository.

Addresses #8898

I did not bother adding death tests with a leading execution space argument.  I think it is good enough as is.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
