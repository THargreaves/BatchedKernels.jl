using TestItems
using TestItemRunner

@run_package_tests

include("validity_tests.jl")
include("throughput_tests.jl")
