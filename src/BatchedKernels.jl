module BatchedKernels

using CUDA
using CUDA: i32
using StaticArrays: @MVector

include("memory.jl")
include("operations.jl")

end
