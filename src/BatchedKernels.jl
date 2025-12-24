module BatchedKernels

using CUDA
using CUDA: i32
using StaticArrays: @MVector

include("memory.jl")
include("operations.jl")

include("fuse/ir.jl")
include("fuse/plan.jl")
include("fuse/emit_expr.jl")
include("fuse/registry.jl")
include("fuse/vmap.jl")

end
