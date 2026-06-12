module BatchedKernels

using CUDA
using CUDA: i32
using LinearAlgebra
using StaticArrays: @MVector
using Base.Broadcast: Broadcasted, BroadcastStyle
import Base.Broadcast

include("containers.jl")
include("memory.jl")
include("operations.jl")

include("fuse/ir.jl")
include("fuse/trace.jl")
include("fuse/overloads.jl")
include("fuse/emit.jl")
include("fuse/plan.jl")
include("fuse/schedule.jl")
include("fuse/output.jl")
include("fuse/codegen.jl")
include("fuse/broadcast.jl")

end
