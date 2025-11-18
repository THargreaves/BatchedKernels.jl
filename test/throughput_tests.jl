using Test


@testitem "Kalman (small, indep)" begin
    using PerformanceTestTools
    PerformanceTestTools.@include("throughput_scripts/kalman_small.jl")
end

@testitem "Matrix Multiplication (small)" begin
    using PerformanceTestTools
    PerformanceTestTools.@include("throughput_scripts/matmul_small.jl")
end