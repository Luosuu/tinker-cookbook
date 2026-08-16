Fix the following issue in the Kokkos repository.

Allows construction of instances with non-blocking initialization occurring on a given execution space instance. Also useful when multiple instances of a device are available, in which case the execution space argument is used to select a device.

For #8032 

I created a unit test that should test the new feature without duplicating any existing tests. However, since it's impossible to compare pools directly, the test is indirect. I'm open to suggestions for better tests.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
