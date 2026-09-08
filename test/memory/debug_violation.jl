# Run in a fresh Julia process with debug_accessors=true: a device assertion
# invalidates the CUDA context, so this must not be included in the normal suite.
using Test
using CUDA
using BatchedKernels
const BK = BatchedKernels
@assert BK.DEBUG_ACCESSORS

function nonuniform_broadcast!(output)
    d = threadIdx().x
    A = BK.RegisterMatrix{Float32}(Val(2), Val(2), Val(32), BK.RowOriented(), Int32(0), d)
    i = isodd(d) ? Int32(1) : Int32(2)
    output[d] = BK.theirs(A, i, Int32(1))
    return nothing
end

output = CUDA.zeros(Float32, 32)
# Compile outside the exception check: compilation failures must not count as
# evidence that the run-time diagnostic detected the invalid collective.
kernel = @cuda launch = false nonuniform_broadcast!(output)
@test_throws CUDA.CUDACore.KernelException CUDA.@sync kernel(output; threads=32)
