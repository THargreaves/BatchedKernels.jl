using TestItems, TestItemRunner, CUDA

# TestItemRunner discovers every @testitem, including the sub-kernel suites.
# CPU-only machines run explicitly tagged algebra/planner/layout tests. Set this
# variable on a GPU machine to exercise the same selection used by hosted CI.
const cpu_only = get(ENV, "BATCHEDKERNELS_TEST_CPU_ONLY", "false") == "true"
if cpu_only || !CUDA.functional()
    @info "Running CPU tests; CUDA kernel tests require a functional GPU"
    @run_package_tests filter = ti -> :cpu in ti.tags
else
    @run_package_tests
end
