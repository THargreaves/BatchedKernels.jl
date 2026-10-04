@testitem "Owned batch storage on host" tags=[:cpu] begin
    using BatchedKernels, LinearAlgebra
    include("storage_cases.jl")
    storage_cases(Array)
end

@testitem "Owned batch storage on CUDA" begin
    using BatchedKernels, CUDA, LinearAlgebra
    CUDA.allowscalar(false)
    include("storage_cases.jl")
    storage_cases(CuArray)
end
