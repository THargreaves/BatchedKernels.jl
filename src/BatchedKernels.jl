module BatchedKernels

using CUDA
using CUDA: i32
using StaticArrays: @MVector

include("cholesky.jl")
include("multiply.jl")

end
