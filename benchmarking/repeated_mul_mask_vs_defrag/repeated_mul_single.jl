using BatchedKernels
using LinearAlgebra
using CUDA
using CUDA: i32
using Random
using JLD2
using Plots
using BenchmarkTools
using Printf

include("repeated_mul_defrag.jl")
include("repeated_mul_mask.jl")

Random.seed!(1234)

nthreads = 256
T = Float32
n_muls = 40
D1 = 4
D = 8

N = Int(ceil(1e9 / (4 * 2 * D1^2)))
nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

M_in_cpu = rand(T, D1, D1, N)
M_in = cu(M_in_cpu)

A_cpu = T.(Matrix(qr(randn(T, D1, D1)).Q))
A = cu(A_cpu)

M_out = CUDA.zeros(T, D1, D1, N)

result_mask = @benchmark begin
    CUDA.@sync @cuda threads=$nthreads blocks=$nblocks $kernel_mul_mask!(
        $M_out, $M_in, $A, Val(Int32($D1)), Val(Int32($D)), Val(Int32($nthreads)), Val($n_muls), Int32($N),
    )
end

result_defrag = @benchmark begin
    CUDA.@sync @cuda threads=$nthreads blocks=$nblocks $kernel_mul_defrag!(
        $M_out, $M_in, $A, Val(Int32($D1)), Val(Int32($D)), Val(Int32($nthreads)), Val($n_muls), Int32($N),
    )
end

# CUDA.@sync @cuda threads=nthreads blocks=nblocks kernel_mul_defrag!(
#     M_out, M_in, A, Val(Int32(D1)), Val(Int32(D)), Val(Int32(nthreads)), Val(n_muls), Int32(N),
# )
# M_out_cpu = Array(M_out)
# # CPU reference
# M_ref = copy(M_in_cpu)
# A_np = A_cpu
# for _ in 1:n_muls
#     for k in axes(M_ref, 3)
#         M_ref[:,:,k] = M_ref[:,:,k] * A_np
#     end
# end

# tol = 1e-3
# for k in axes(M_ref, 3)
#     e = maximum(abs.(M_out_cpu[:,:,k] .- M_ref[:,:,k]))
#     if e > tol
#         println("first bad k=$k, error=$e")
#         println("GPU:\n", M_out_cpu[:,:,k])
#         println("CPU:\n", M_ref[:,:,k])
#         break
#     end
# end

# err = maximum(abs.(M_out_cpu .- M_ref))
# println("max error: $err")

ratio = median(result_mask.times) / median(result_defrag.times)
println("ratio=$ratio, mask_time=$(median(result_mask.times)), defrag_time=$(median(result_defrag.times))")