module BatchedKernels

using CUDA
using CUDA: i32
using LinearAlgebra
using StaticArrays: @MVector
using Base.Broadcast: Broadcasted, BroadcastStyle
import Base.Broadcast

include("containers.jl")
include("memory.jl")
include("accessors.jl")
include("operations.jl")
include("variant_elementwise.jl")
include("variant_factorizations.jl")
include("variant_vectors.jl")
include("covariance.jl")
include("block_qr.jl")
include("variant_block_qr.jl")
include("variant_block_qr_columns.jl")

include("fuse/ir.jl")
include("fuse/trace.jl")
include("fuse/overloads.jl")
include("fuse/block_qr.jl")
include("fuse/emit.jl")
include("fuse/variants.jl")
include("fuse/plan.jl")
include("fuse/assignment.jl")
include("fuse/schedule.jl")
include("fuse/automatic.jl")
include("fuse/output.jl")
include("fuse/codegen.jl")
include("fuse/broadcast.jl")

end
