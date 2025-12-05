using StaticArrays
using BenchmarkTools
using LinearAlgebra
using CUDA
using CUDA: i32

include(joinpath("..", "forward_solve", "magma_non_strided.jl"))
include(joinpath("..", "cholesky", "magma_non_strided.jl"))
include(joinpath("..", "matmul", "magma_non_strided.jl"))

function magma_non_strided_kalman!(
    dPo, dPi, dA, dQ, dH, dR, dB, info_d, D, N, queue_ptr,
)# where {T}
    L = Magma.LibMagma.MagmaLeft
    R = Magma.LibMagma.MagmaRight
    LO = Magma.LibMagma.MagmaLower
    NT = Magma.LibMagma.MagmaNoTrans
    TR = Magma.LibMagma.MagmaTrans
    NUNIT = Magma.LibMagma.MagmaNonUnit

    zero = 0.0f0
    one = 1.0f0

    # Temp = A * Pin, store in dB
    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        one, dA, D,
        dPi, D,
        zero, dB, D,
        N,
        queue_ptr[],
    )

    # P_pred = temp * A' => Q <- dB * A' + Q
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dB, D,  # temp = A * Pin
        dA,  D,  # A'
        one, dQ, D,   # P_pred
        N,
        queue_ptr[],
    )
    # Now dQ contains P_pred

    # dB <- H * P_pred
    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        one, dH, D,
        dQ, D,  # P_pred
        zero, dB, D,
        N,
        queue_ptr[],
    )

    # dR <- H * P_pred * H' + R
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dB, D,  # H * P_pred
        dH, D,
        one, dR, D,
        N,
        queue_ptr[],
    )
    # Now dR contains S

    # In-place cholesky for S = dR = H * P_pred * H' + R
    magma_spotrf_batched!(
        LO,
        D,
        dR, D,
        info_d,
        N,
        queue_ptr[],
    )
    # dR now contains L, where S = LL^T

    # Temp: dPi <- P_pred * H'
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dQ, D,  # P_pred
        dH, D,
        zero, dPi, D,
        N,
        queue_ptr[],
    )

    # Temp: dPi <- (P_pred * H') \ S (S=dR)
    magma_lower_solve_wrapper!(
        R, LO, TR, NUNIT,
        D, D,
        one,
        dR, D,  # L
        dPi, D,  # P_pred * H'
        N,
        queue_ptr[],
    )

    # Solve K = <- (P_pred * H') \ S (S=dR), stored in dPi
    magma_lower_solve_wrapper!(
        R, LO, NT, NUNIT,
        D, D,
        one, dR, D,  # L
        dPi, D,
        N,
        queue_ptr[],
    )
    # Now dPi contains K

    # dB <- K * L
    magmablas_sgemm_batched!(
        NT, NT,
        D, D, D,
        one, dPi, D,  # K
        dR, D,  # S
        zero, dB, D,
        N,
        queue_ptr[],
    )

    # dPo <- (K*L) * (K*L)^T = K LL^T K^T = K * S * K'
    magmablas_sgemm_batched!(
        NT, TR,
        D, D, D,
        one, dB, D,
        dB, D,
        zero, dPo, D,
        N,
        queue_ptr[],
    )

    # Answer is P_pred - P_new, i.e dQ - dPo
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
end

@inline function unsafe_strided_batch_repeat(strided::DenseCuArray{T}, N::Int) where {T}
    batch_size = N
    #ptrs = [pointer(strided, (i-1)*batch_stride + 1) for i in 1:batch_size]
    # fill the array on the GPU to avoid synchronous copies and support larger batch sizes
    ptrs = CuArray{CuPtr{T}}(undef, batch_size)
    # device-side code
    ## COV_EXCL_START
    function compute_pointers()
        i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        grid_stride = gridDim().x * blockDim().x
        while i <= length(ptrs)
            @inbounds ptrs[i] =
                reinterpret(CuPtr{T}, pointer(strided, 1i32))
            i += grid_stride
        end
        return
    end
    ## COV_EXCL_STOP
    kernel = @cuda launch = false compute_pointers()
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, batch_size)
    blocks = min(config.blocks, cld(batch_size, threads))
    @cuda threads blocks compute_pointers()
    return ptrs
end

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, queue_ptr, ::Val{:magma_non_strided})
    D, _, N = size(P_in_cpu)

    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)
    B = CUDA.zeros(D, D, N)

    dPo = CUDA.CUBLAS.unsafe_strided_batch(P_out)
    dPi = CUDA.CUBLAS.unsafe_strided_batch(P_in)

    dA = unsafe_strided_batch_repeat(A, N)
    dQ = unsafe_strided_batch_repeat(Q, N)
    dH = unsafe_strided_batch_repeat(H, N)
    dR = unsafe_strided_batch_repeat(R, N)
    dB = CUDA.CUBLAS.unsafe_strided_batch(B)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    bench_results = @benchmark begin
        magma_non_strided_kalman!($dPo, $dPi, $dA, $dQ, $dH, $dR, $dB, $info_d, $D, $N, $queue_ptr)
    end

    return median(bench_results.times) / 1e9 / N
end
