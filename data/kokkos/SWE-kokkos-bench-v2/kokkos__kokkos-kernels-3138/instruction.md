Fix the following issue in the Kokkos repository.

Extend `KokkosBatched` Iamax with Team and TeamVector variants and unify them with the existing Serial interface. `SerialIamax::invoke(x)`, `TeamIamax<MemberType>::invoke(member, x)`, and `TeamVectorIamax<MemberType>::invoke(member, x)` must accept a one-dimensional view and return its `size_type`. Return the index of the first element having the largest absolute value; return 0 when the view has at most one element. Use the scalar type's appropriate absolute-value behavior. The Team and TeamVector forms must perform the reduction over execution-appropriate team or vector ranges and preserve first-location tie breaking.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
