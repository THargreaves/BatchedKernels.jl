using CUDA
using Test
using CUDA: i32
using LinearAlgebra
using BatchedKernels
using Random

include("../ours.jl")

nthreads = 2^8
N = 2^10 + 113
THRESH = 10

function cpu_kalman_cov(P, A, Q, H, R)
    P_pred = A * P * A' + Q
    S = H * P_pred * H' + R
    K = P_pred * H' / S
    P = P_pred - K * S * K'
    return P
end

for D in 2:16
    Random.seed!(1234)
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    A_elem = rand(Float32, D, D) / Float32(D)
    Q_elem = rand(Float32, D, D) / Float32(D)^2
    Q_elem = Q_elem * Q_elem' + 0.01f0 * I

    H_elem = rand(Float32, D, D) / Float32(D)
    R_elem = rand(Float32, D, D) / Float32(D)^2
    R_elem = R_elem * R_elem' + 0.01f0 * I

    S_Q_elem = Float32.(Matrix(cholesky(Q_elem).L))
    S_R_elem = Float32.(Matrix(cholesky(R_elem).L))

    # Generate batched S = cholesky(P).L
    S_cpu = Array{Float32}(undef, D, D, N)
    P_cpu = Array{Float32}(undef, D, D, N)
    for i in 1:N
        P_i = rand(Float32, D, D) / Float32(D)
        P_i = P_i * P_i' + 0.1f0 * I
        P_cpu[:, :, i] = P_i
        S_cpu[:, :, i] = Float32.(Matrix(cholesky(P_i).L))
    end

    S_in = CuArray(S_cpu)
    A_gpu = CuArray(A_elem)
    S_Q_gpu = CuArray(S_Q_elem)
    H_gpu = CuArray(H_elem)
    S_R_gpu = CuArray(S_R_elem)

    S_out = CuArray{Float32}(undef, D, D, N)

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval = div(32, D & -D) * D

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval
    shmem_bytes = (2 * shmem_elems + 4 * shmem_size_fixed) * sizeof(Float32)

    CUDA.@sync @cuda threads = nthreads blocks = nblocks shmem = shmem_bytes kernel_sqrt_kalman!(
        S_out,
        S_in,
        A_gpu,
        S_Q_gpu,
        H_gpu,
        S_R_gpu,
        Val(Int32(D)),
        Val(Int32(THRESH)),
        Val(Int32(nthreads)),
        Int32(N),
        Val(:small),
    )

    S_out_cpu = Array(S_out)
    max_error = 0.0

    for i in 1:N
        P_new_ref = cpu_kalman_cov(P_cpu[:, :, i], A_elem, Q_elem, H_elem, R_elem)
        P_new_sqrt = S_out_cpu[:, :, i] * S_out_cpu[:, :, i]'

        err = maximum(abs.(P_new_ref - P_new_sqrt))
        max_error = max(max_error, err)
        if max_error > 1e-4
            println("error at i=$i for D=$D: $max_error")
            break
        end
    end

    println("D=$D, max_error=$max_error")
    @test max_error < 1e-4
end