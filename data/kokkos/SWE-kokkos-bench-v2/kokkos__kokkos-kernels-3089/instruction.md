Fix the following issue in the Kokkos repository.

Convert the batched dense rotation (`rotm`) APIs to match `Rotmg` by making `Flag` a runtime parameter instead of a compile-time template argument. Remove the `Flag` template parameter from `SerialRotm`, `TeamRotm`, and `TeamVectorRotm`. The parameter vector must have length 5, with `param[0]` holding the flag and `param[1..4]` holding `h11`, `h21`, `h12`, and `h22`. Valid flags are `-2` (identity/no-op), `-1`, `0`, and `1`; any other value must make `invoke` return `1`. Read the flag at runtime and apply the rotation behavior consistent with `Rotmg`.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
