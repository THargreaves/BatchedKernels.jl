using TestItems, TestItemRunner, CUDA, Test

# Cold CUDA compilation can keep a single test item busy for minutes, and the
# default test set reports nothing until the whole run ends. Print each package,
# file and test item as it starts and finishes so slow items can be identified.
struct ProgressTestSet <: Test.AbstractTestSet
    inner::Test.DefaultTestSet
end
function ProgressTestSet(desc; kwargs...)
    indent = "  "^Test.get_testset_depth()
    println(Libc.strftime("[%H:%M:%S] ", time()), indent, "start ", desc)
    flush(stdout)
    return ProgressTestSet(Test.DefaultTestSet(desc; kwargs...))
end
Test.record(ts::ProgressTestSet, result) = Test.record(ts.inner, result)
function Test.finish(ts::ProgressTestSet)
    elapsed = round(time() - ts.inner.time_start; digits=1)
    indent = "  "^Test.get_testset_depth()
    println(
        Libc.strftime("[%H:%M:%S] ", time()),
        indent,
        "done  ",
        ts.inner.description,
        " ($(elapsed) s)",
    )
    flush(stdout)
    Test.finish(ts.inner)
    return ts
end

# Discover only test/: repository-wide discovery also executes nested worktrees.
# Each test item explicitly imports BatchedKernels.
# CPU-only machines run explicitly tagged algebra/planner/layout tests. Set
# CPU_ONLY on a GPU machine for the hosted CI selection; EXTENDED=false omits
# the separately tagged large-shape cases. Full coverage remains the default.
const extended = get(ENV, "BATCHEDKERNELS_TEST_EXTENDED", "true") != "false"
const include_item = ti -> extended || !(:extended in ti.tags)
const cpu_only = get(ENV, "BATCHEDKERNELS_TEST_CPU_ONLY", "false") == "true"
if cpu_only || !CUDA.functional()
    @info "Running CPU tests; CUDA kernel tests require a functional GPU"
    TestItemRunner.run_tests(
        @__DIR__; filter=ti -> :cpu in ti.tags && include_item(ti), testset=ProgressTestSet
    )
else
    TestItemRunner.run_tests(@__DIR__; filter=include_item, testset=ProgressTestSet)
end
