Fix the following issue in the Kokkos repository.

Fix `ScatterValue` so that moving does not leave the source referencing the same underlying storage. The class currently declares only a move constructor that copies its internal reference, so the moved-from object remains linked to the moved-to object. For every specialization (e.g., `ScatterSum`, `ScatterProd`, `ScatterMin`, `ScatterMax` with atomic and non-atomic access), remove the erroneous move constructor, explicitly delete the copy constructor and copy assignment operator, and add an explicitly defaulted destructor. This suppresses implicit move operations and ensures correct special-member control on device.

Remove the unused `join` member. Since `join` is invoked from the arithmetic, increment, decrement, and compound-assignment operators as well as from `update()`, eliminate those call sites and perform the corresponding updates directly, using atomic operations where the access mode requires them.

Drive-by: remove the explicit default destructor from `ScatterAccess`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
