Fix the following issue in the Kokkos repository.

To align with [Rotmg](#3088), we need to make `Flag` a runtime parameter included in param variable.
This way we can use them like

[implementation suggestion omitted]

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
