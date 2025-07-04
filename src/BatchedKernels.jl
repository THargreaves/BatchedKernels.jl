module BatchedKernels

using CUDA
using CUDA: i32

include("cholesky.jl")
include("multiply.jl")
include("qr.jl")

end
