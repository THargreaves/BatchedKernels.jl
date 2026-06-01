using Random
using LinearAlgebra

include("../../config/Schedule.jl")
include("../repeated_mul_mask.jl")

Random.seed!(1234)

T = Float32
Dx = 4
Dy = 8
D = max(Dx, Dy)
# N = Int(ceil(1e9 / (4 * 2 * D1^2)))
N = 2^9 + 1
n_muls = 10
nthreads = Schedule.best_nthreads("matmul", D)

M_in_cpu = rand(T, Dx, Dy, N)
A_cpu = rand(T, Dy, Dx)
M_out_cpu = zeros(T, Dx, Dx, N)
    
Ms_out = cu(M_out_cpu)
Ms_in = cu(M_in_cpu)
A = cu(A_cpu)

nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))
CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_mul_mask!(
    Ms_out, Ms_in, A,
    Val(Int32(Dx)), Val(Int32(Dy)), Val(Int32(D)),
    Val(Int32(nthreads)), Val(Int32(n_muls)), Int32(N),
)
path = joinpath(@__DIR__, "mask.ll")
open(path, "w") do io
    CUDA.@device_code_llvm io=io @cuda launch=false kernel_mul_mask!(
        Ms_out, Ms_in, A,
        Val(Int32(Dx)), Val(Int32(Dy)), Val(Int32(D)),
        Val(Int32(nthreads)), Val(Int32(n_muls)), Int32(N),
    )
end
M_res_cpu = Array(Ms_out)

for i in 1:N
    M_ref = M_in_cpu[:, :, i] * A_cpu
    M_out_cpu[:, :, i] = M_ref
    error = maximum(abs.(M_res_cpu[:, :, i] .- M_ref))
    if error > 1e-5
        println("error at i=$i, error=$error")
        dislpay(M_ref)
        display(M_res_cpu[:, :, i])
        break
    end
end

max_error = maximum(abs.(M_res_cpu .- M_out_cpu))
println("finished, max_error=$max_error")