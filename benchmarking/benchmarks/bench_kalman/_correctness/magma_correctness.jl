using CUDA
using CUDA: i32
using LinearAlgebra
using Magma

# Include both implementations
include("../magma.jl")   # has kalman_magma!, magmablas_sgemm_batched!, etc.
include("../ours.jl")    # has kernel_kalman!, BatchedKernels, etc.

function test_kalman_correctness(; D=4, N=128, T=Float32, atol=1e-3, rtol=1e-3)
    # ── Generate inputs (same as benchmark caller) ──
    P_in_cpu = zeros(T, D, D, N)
    for i in 1:N
        P_i = rand(T, D, D) / T(D)
        P_i = P_i * P_i' + 0.1f0 * I
        P_in_cpu[:, :, i] = P_i
    end

    A_cpu = rand(T, D, D) / T(D)

    Q_elem = rand(T, D, D) / T(D)^2
    Q_cpu = Q_elem * Q_elem' + 0.01f0 * I

    H_cpu = rand(T, D, D) / T(D)

    R_elem = rand(T, D, D) / T(D)^2
    R_cpu = R_elem * R_elem' + 0.01f0 * I

    # ── Run "ours" ──
    P_out_ours = CUDA.zeros(T, D, D, N)
    P_in_ours = cu(P_in_cpu)
    A_d = cu(A_cpu)
    Q_d = cu(Q_cpu)
    H_d = cu(H_cpu)
    R_d = cu(R_cpu)

    nthreads = 256
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval = div(32, D & -D) * D
    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval
    shmem_bytes = sizeof(T) * (3 * shmem_elems + 4 * shmem_size_fixed)

    kernel = @cuda launch=false kernel_kalman!(
        P_out_ours, P_in_ours, A_d, Q_d, H_d, R_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        P_out_ours, P_in_ours, A_d, Q_d, H_d, R_d,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )
    result_ours = Array(P_out_ours)

    # ── Run MAGMA ──
    P_out_magma = CUDA.zeros(T, D, D, N)
    P_in_magma = cu(P_in_cpu)
    W_magma = CUDA.zeros(T, D, D, N)

    dPo = CUDA.CUBLAS.unsafe_strided_batch(P_out_magma)
    dPi = CUDA.CUBLAS.unsafe_strided_batch(P_in_magma)
    dW = CUDA.CUBLAS.unsafe_strided_batch(W_magma)
    dA = unsafe_strided_batch_repeat(A_d, N)
    dH = unsafe_strided_batch_repeat(H_d, N)
    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    Magma.LibMagma.magma_init()
    queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
    device = 0
    Magma.LibMagma.magma_queue_create_internal(
        device,
        queue_ptr,
        C_NULL,  # func
        C_NULL,  # file
        0,       # line
    )

    kalman_magma!(
        dPo, dPi, dW, dA, dH,
        reshape(P_out_magma, :), reshape(P_in_magma, :), reshape(W_magma, :),
        reshape(Q_d, :), reshape(R_d, :),
        info_d, D, N, queue_ptr,
    )
    CUDA.synchronize()
    result_magma = Array(P_out_magma)

    # ── Compare ──
    max_err = 0.0f0
    n_bad = 0
    for i in 1:N
        err = maximum(abs.(result_ours[:, :, i] .- result_magma[:, :, i]))
        max_err = max(max_err, err)
        if !isapprox(result_ours[:, :, i], result_magma[:, :, i]; atol, rtol)
            n_bad += 1
            if n_bad <= 3  # print first few mismatches
                println("Mismatch at batch $i:")
                println("  ours:  ", result_ours[:, :, i])
                println("  magma: ", result_magma[:, :, i])
                println("  max elem diff: $err")
            end
        end
    end

    println("\nD=$D, N=$N")
    println("Max element-wise error: $max_err")
    println("Mismatched batches: $n_bad / $N (atol=$atol, rtol=$rtol)")
    if n_bad == 0
        println("✓ PASS")
    else
        println("✗ FAIL")
    end

    return n_bad == 0
end

# Run for a few sizes
for D in 2:16
    test_kalman_correctness(D=D, N=256)
    println()
end