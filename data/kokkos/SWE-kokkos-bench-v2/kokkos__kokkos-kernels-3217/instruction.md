Fix the following issue in the Kokkos repository.

In KokkosBatched, extend the Norm enumeration and align all norm computations with the BLAS 1-norm family.

Public API: Add Norm::GenuineLInf. Correct Norm::LInf.

For any 1-D view X (real or complex float/double), across Serial, Team, and TeamVector modes:
- Norm::L1: sum_i (|Re(x_i)| + |Im(x_i)|) [BLAS SASUM/SCASUM/DASUM/DZASUM]
- Norm::GenuineL1: sum_i sqrt(|Re(x_i)|^2 + |Im(x_i)|^2) [BLAS SCSUM1/DZSUM1]
- Norm::L2 / Norm::ScaledL2: sqrt(sum_i (|Re(x_i)|^2 + |Im(x_i)|^2)); ScaledL2 avoids overflow via scaling [BLAS SNRM2/SCNRM2/DNRM2/DZNRM2]
- Norm::LInf: max_i (|Re(x_i)| + |Im(x_i)|) [BLAS ISAMAX/ICAMAX/IDAMAX/IZAMAX]. This intentionally changes the prior max-modulus behavior.
- Norm::GenuineLInf: max_i sqrt(|Re(x_i)|^2 + |Im(x_i)|^2) [BLAS ICMAX1/IZMAX1]. This preserves the previous Norm::LInf result.

For real inputs, imaginary parts are zero and formulas reduce to standard real norms.

Compatibility: L1, GenuineL1, L2, and ScaledL2 must keep their current definitions. The LInf change is required; GenuineLInf must be available so max-modulus remains accessible. Support must cover float, double, complex<float>, and complex<double>. Public documentation should list all six norms with these BLAS-aligned definitions.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
