Fix the following issue in the Kokkos repository.

Promote Kokkos_ScopeGuard.hpp and Kokkos_InitializeFinalize.hpp to public headers. Kokkos_InitializeFinalize.hpp must expose only initialization/finalization symbols (initialize, finalize, is_initialized, is_finalized, push_finalize_hook, and internal pre_finalize/post_finalize). It must not contain fence, print_configuration, device_id, num_devices, num_threads, show_warnings, tune_internals, or declare_configuration_metadata. Move those remaining declarations into separate new internal-only headers. Update Kokkos_Core.hpp to continue providing the full legacy API by including both promoted public headers plus the new internal split headers, replacing only the old internal includes for the moved headers; leave all other includes and behavior intact. Update Kokkos_ScopeGuard.hpp to include the public <Kokkos_InitializeFinalize.hpp>. Update all other source references from the old internal path for these headers to the new public paths. Do NOT rename Kokkos_Core.cpp. Preserve backward compatibility so that code including Kokkos_Core.hpp works unchanged. Add a changelog entry.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
