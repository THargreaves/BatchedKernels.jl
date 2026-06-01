using Test
using BatchedKernels
using CUDA
using CUDA: i32
using Random
using LinearAlgebra

include("../original_kalman.jl")

function kalman_cov(P, A, Q, H, R)
    # Predict step
    P_pred = A * P * A' + Q

    # Kalman gain
    S = H * P_pred * H' + R
    K = P_pred * H' / Symmetric(S)

    # Update step
    I_KH = I - K * H
    
    P_new = I_KH * P_pred

    return P_new
end

T = Float32
N = 2^9 + 1
nthreads = 128

for D in 9:32
    Random.seed!(1234)
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes = sizeof(Float32) * 5 * shmem_elems

    A_cpu = rand(T, D, D, N) / T(D)
    Q_cpu = zeros(T, D, D, N)
    for i in 1:N
        Q_elem = rand(T, D, D) / T(D)^2
        Q_cpu[:, :, i] = Q_elem * Q_elem' + 0.01f0 * I
    end

    H_cpu = rand(T, D, D, N) / T(D)
    R_cpu = zeros(T, D, D, N)
    for i in 1:N
        R_elem = rand(T, D, D) / T(D)^2
        R_cpu[:, :, i] = R_elem * R_elem' + 0.01f0 * I
    end

    P_in_cpu = Array{T}(undef, D, D, N)
    for i in 1:N
        P_i = rand(T, D, D) / T(D)
        P_i = P_i * P_i' + 0.1f0 * I
        P_in_cpu[:, :, i] = P_i
    end

    P_out_cpu = zeros(T, D, D, N)

    Ps_out = cu(P_out_cpu)
    Ps_in = cu(P_in_cpu)
    As = cu(A_cpu)
    Qs = cu(Q_cpu)
    Hs = cu(H_cpu)
    Rs = cu(R_cpu)

    kernel = @cuda launch = false kernel_kalman_orig!(
        Ps_out, Ps_in, As, Qs, Hs, Rs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        Ps_out, Ps_in, As, Qs, Hs, Rs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
        threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
    )

    Ps_out_res = Array(Ps_out)

    Ps_out_ref = zeros(T, D, D, N)
    for i in 1:N
        Ps_out_ref[:, :, i] = kalman_cov(P_in_cpu[:, :, i], A_cpu[:, :,  i], Q_cpu[:, :, i], H_cpu[:, :, i], R_cpu[:, :, i])
    end
    max_error = maximum(abs.(Ps_out_ref .- Ps_out_res))

    println("D=$D, max_error=$max_error")

    @test max_error < 1e-3
end