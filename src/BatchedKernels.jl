module BatchedKernels

using CUDA

include("cholesky.jl")
include("multiply.jl")
include("qr.jl")

end
