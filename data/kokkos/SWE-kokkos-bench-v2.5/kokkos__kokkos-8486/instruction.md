Fix the following issue in the Kokkos repository.

Modernize Kokkos::Experimental::ErrorReporter, keeping it in the Experimental namespace. Replace DualView usage with Kokkos::View. Add a default template parameter DeviceType = DefaultExecutionSpace::device_type. Rename public members to snake_case: capacity(), num_reports(), num_report_attempts(), full(), clear(), resize(), add_report(), and get_reports(). Add constructors ErrorReporter(const std::string& label, int max_results) and ErrorReporter(int max_results), with the latter defaulting the label to "ErrorReporter". The const member get_reports() must return std::pair<std::vector<int>, std::vector<report_type>> with size equal to min(num_report_attempts(), capacity()). Deprecate the old camelCase names—getCapacity, getNumReports, getNumReportAttempts, and getReports overloads accepting std::vector or host mirror Views—under KOKKOS_ENABLE_DEPRECATED_CODE_4. Implement clear() to reset the attempt counter to 0. Implement full() as attempts >= capacity. For resize(): growing beyond the prior capacity resets attempts to num_reports() only when prior attempts exceeded the old capacity; shrinking clamps visible reports but does not reset attempts.

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
