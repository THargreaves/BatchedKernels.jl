using CUDA
using BatchedKernels
using Random
using Test

include("../kalman_defrag.jl")

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
Dx_min = 2
Dx_max = 8
Dy_min = 2
Dy_max = 8

for Dx in Dx_min:Dx_max
    for Dy in Dy_min:Dy_max
        Random.seed!(1234)
        nthreads = 128  # Limited by the shmem buffers in defrag
        D = max(Dx, Dy)
        N = 2^9 + 1

        P_in = Array{T}(undef, Dx, Dx, N)
        for i in 1:N
            P_i = rand(T, Dx, Dx) / T(Dx)
            P_i = P_i * P_i' + 0.1f0 * I
            P_in[:, :, i] = P_i
        end

        A_cpu = rand(Float32, Dx, Dx) / Float32(Dx)

        Q_cpu = rand(Float32, Dx, Dx) / Float32(Dx)^2
        Q_cpu = Q_cpu * Q_cpu' + 0.01f0 * I

        H_cpu = rand(T, Dy, Dx) / T(D)

        R_cpu = rand(T, Dy, Dy) / T(Dy)^2
        R_cpu = R_cpu * R_cpu' + 0.01f0 * I

        P_out = zeros(T, Dx, Dx, N)

        Ps_out = cu(P_out)
        Ps_in = cu(P_in)
        A = cu(A_cpu)
        Q = cu(Q_cpu)
        H = cu(H_cpu)
        R = cu(R_cpu)

        nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

        n_mats_per_warp = 32 ÷ D
        n_warps = nthreads ÷ 32
        padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
        pad_interval = div(32, D & -D) * D

        shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
        # Use D to compute this instead of the true shmem_size_fixed used in the kernel for both simplicity and equality of comparison against masking
        shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval

        shmem_bytes = sizeof(Float32) * (
            4 * shmem_elems + 4 * shmem_size_fixed
        )

        kernel = @cuda launch=false kernel_kalman_defrag!(
            Ps_out, Ps_in, A, Q, H, R,
            Val(Int32(Dx)), Val(Int32(Dy)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
        )
        CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

        CUDA.@sync kernel(
            Ps_out, Ps_in, A, Q, H, R,
            Val(Int32(Dx)), Val(Int32(Dy)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N);
            threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
        )
        Ps_out_res = Array(Ps_out)

        Ps_out_ref = zeros(T, Dx, Dx, N)
        for i in 1:N
            Ps_out_ref[:, :, i] = kalman_cov(P_in[:, :, i], A_cpu, Q_cpu, H_cpu, R_cpu)
        end
        max_error = maximum(abs.(Ps_out_ref .- Ps_out_res))

        println("Dx=$Dx, Dy=$Dy, max_error=$max_error")

        @test max_error < 1e-3
    end
end