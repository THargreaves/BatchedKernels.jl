using CUDA
using CUDA: i32
using BenchmarkTools

function magma_spotrf_batched!(
    uplo::Magma.LibMagma.magma_uplo_t,
    n::Integer,
    dA,
    lda::Integer,
    info_array,
    batchCount::Integer,
    queue::Magma.LibMagma.magma_queue_t,
)
    return ccall(
        (:magma_spotrf_batched, Magma.LibMagma.libmagma),
        Cvoid,
        (
            Magma.LibMagma.magma_uplo_t,
            Magma.LibMagma.magma_int_t,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            CuPtr{Magma.LibMagma.magma_int_t},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_queue_t,
        ),
        uplo,
        n,
        dA,
        lda,
        info_array,
        batchCount,
        queue
    )
end

function magmablas_strsm_batched!(
    side::Magma.LibMagma.magma_side_t,
    uplo::Magma.LibMagma.magma_uplo_t,
    transA::Magma.LibMagma.magma_trans_t,
    diag::Magma.LibMagma.magma_diag_t,
    m::Integer,
    n::Integer,
    alpha::Cfloat,
    dA_array,
    ldda::Integer,
    dB_array,
    lddb::Integer,
    batchCount::Integer,
    queue,
)
    ccall(
        (:magmablas_strsm_batched,Magma.LibMagma.libmagma),
        Cvoid,
        (
            Magma.LibMagma.magma_side_t,
            Magma.LibMagma.magma_uplo_t,
            Magma.LibMagma.magma_trans_t,
            Magma.LibMagma.magma_diag_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_queue_t
        ),
        side,
        uplo,
        transA,
        diag,
        m,
        n,
        alpha,
        dA_array,
        ldda,
        dB_array,
        lddb,
        batchCount,
        queue,
    )
    return nothing
end

function magmablas_sgemm_batched!(
    transA,
    transB,
    m,
    n,
    k,
    alpha,
    dA,
    ldda,
    dB,
    lddb,
    beta,
    dC,
    lddc,
    batchCount,
    queue,
)
    return ccall(
        (:magmablas_sgemm_batched, Magma.LibMagma.libmagma),
        Cvoid,
        (
            Magma.LibMagma.magma_trans_t,
            Magma.LibMagma.magma_trans_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            Cfloat,
            CuPtr{CuPtr{Cfloat}},
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_int_t,
            Magma.LibMagma.magma_queue_t,
        ),
        transA,
        transB,
        m,
        n,
        k,
        alpha,
        dA,
        ldda,
        dB,
        lddb,
        beta,
        dC,
        lddc,
        batchCount,
        queue,
    )
end

# dst[i] += src[((i-1) % DD) + 1]: adds a shared D×D matrix into each batch slot
function _add_shared_kernel!(dst, src, DD, total)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= total
        @inbounds dst[i] += src[((i - 1i32) % DD) + 1i32]
    end
    return
end

# dst[i] -= src[i]
function _sub_batched_kernel!(dst, src, total)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= total
        @inbounds dst[i] -= src[i]
    end
    return
end

function add_shared!(dst::DenseCuArray{Float32}, src::DenseCuArray{Float32}, DD::Int, N::Int)
    total = DD * N
    nthreads = 256
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _add_shared_kernel!(dst, src, Int32(DD), Int32(total))
end

function sub_batched!(dst::DenseCuArray{Float32}, src::DenseCuArray{Float32}, total::Int)
    nthreads = 256
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _sub_batched_kernel!(dst, src, Int32(total))
end

@inline function unsafe_strided_batch_repeat(strided::DenseCuArray{T}, N::Int) where {T}
    ptrs = CuArray{CuPtr{T}}(undef, N)
    function _fill_ptrs()
        i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        grid_stride = gridDim().x * blockDim().x
        while i <= length(ptrs)
            @inbounds ptrs[i] = reinterpret(CuPtr{T}, pointer(strided, 1i32))
            i += grid_stride
        end
        return
    end
    kernel = @cuda launch=false _fill_ptrs()
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, N)
    blocks = min(config.blocks, cld(N, threads))
    @cuda threads blocks _fill_ptrs()
    return ptrs
end

# ─── Main Kalman covariance update ──────────────────────────────
#
#  Slot usage:   P_in          P_out         W
# ──────────────────────────────────────────────────────────────────
#  1.            P_in          -             A*P_in
#  2.            (free)        A*P*A'        -
#  3.            (free)        P_pred        -
#  4.            P_pred*H'     P_pred        -
#  5.            P_pred*H'     P_pred        H*P_pred*H'
#  6.            P_pred*H'     P_pred        S
#  7.            P_pred*H'     P_pred        L
#  8-9.          K             P_pred        L (no longer needed)
#  10.           K             P_pred        K*H
#  11.           K*H*P_pred    P_pred        -
#  12.           -             (I-K*H)*P     -

function kalman_magma!(
    dPo, dPi, dW,       # batched pointer arrays
    dA, dH,              # shared pointer arrays (all → same matrix)
    P_out, P_in, W,      # flat CuArrays for custom kernels
    Q, R,                # shared D×D flat CuArrays (read-only)
    info_d, D, N, queue_ptr,
)
    NT = Magma.LibMagma.MagmaNoTrans
    TR = Magma.LibMagma.MagmaTrans
    LO = Magma.LibMagma.MagmaLower
    RI = Magma.LibMagma.MagmaRight
    NUNIT = Magma.LibMagma.MagmaNonUnit

    zero = 0.0f0
    one = 1.0f0
    DD = D * D

    # 1. W = A * P_in
    magmablas_sgemm_batched!(
        NT, NT, D, D, D,
        one, dA, D, dPi, D,
        zero, dW, D, N, queue_ptr[],
    )

    # 2. P_out = W * A'
    magmablas_sgemm_batched!(
        NT, TR, D, D, D,
        one, dW, D, dA, D,
        zero, dPo, D, N, queue_ptr[],
    )

    # Sync MAGMA queue before custom kernel on default stream
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)

    # 3. P_out += Q  →  P_out = P_pred
    add_shared!(P_out, Q, DD, N)
    CUDA.synchronize()

    # 4. P_in = P_pred * H'
    magmablas_sgemm_batched!(
        NT, TR, D, D, D,
        one, dPo, D, dH, D,
        zero, dPi, D, N, queue_ptr[],
    )

    # 5. W = H * (P_pred * H') = H * P_pred * H'
    magmablas_sgemm_batched!(
        NT, NT, D, D, D,
        one, dH, D, dPi, D,
        zero, dW, D, N, queue_ptr[],
    )

    # Sync before custom kernel
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)

    # 6. W += R  →  W = S
    add_shared!(W, R, DD, N)
    CUDA.synchronize()

    # 7. Cholesky: S = L*L'  (L overwrites W)
    magma_spotrf_batched!(
        LO, D, dW, D, info_d, N, queue_ptr[],
    )

    # 8. trsm: P_in = P_in * L^{-T}  (P_in holds P_pred*H')
    magmablas_strsm_batched!(
        RI, LO, TR, NUNIT,
        D, D, one,
        dW, D, dPi, D,
        N, queue_ptr[],
    )

    # 9. trsm: P_in = P_in * L^{-1}  →  P_in = K
    magmablas_strsm_batched!(
        RI, LO, NT, NUNIT,
        D, D, one,
        dW, D, dPi, D,
        N, queue_ptr[],
    )

    # 10. W = K * H
    magmablas_sgemm_batched!(
        NT, NT, D, D, D,
        one, dPi, D, dH, D,
        zero, dW, D, N, queue_ptr[],
    )

    # 11. P_in = (K*H) * P_pred = W * P_out
    magmablas_sgemm_batched!(
        NT, NT, D, D, D,
        one, dW, D, dPo, D,
        zero, dPi, D, N, queue_ptr[],
    )

    # Sync before custom kernel
    Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)

    # 12. P_out -= P_in  →  P_out = P_pred - K*H*P_pred = (I - K*H)*P_pred
    sub_batched!(P_out, P_in, DD * N)

    CUDA.synchronize()
end

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, queue_ptr, _, ::Val{:magma})
    D, _, N = size(P_in_cpu)

    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)
    W = CUDA.zeros(Float32, D, D, N)

    dPo = CUDA.CUBLAS.unsafe_strided_batch(P_out)
    dPi = CUDA.CUBLAS.unsafe_strided_batch(P_in)
    dW = CUDA.CUBLAS.unsafe_strided_batch(W)

    dA = unsafe_strided_batch_repeat(A, N)
    dH = unsafe_strided_batch_repeat(H, N)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    # Flat views into contiguous GPU memory for custom kernels
    P_out_flat = reshape(P_out, :)
    P_in_flat = reshape(P_in, :)
    W_flat = reshape(W, :)
    Q_flat = reshape(Q, :)
    R_flat = reshape(R, :)

    bench_results = @benchmark begin
        kalman_magma!(
            $dPo,
            $dPi,
            $dW,
            $dA,
            $dH,
            $P_out_flat,
            $P_in_flat,
            $W_flat,
            $Q_flat,
            $R_flat,
            $info_d,
            $D,
            $N,
            $queue_ptr,
        )
    end setup=begin
        copyto!($P_out, $P_out_cpu)
        copyto!($P_in, $P_in_cpu)
        fill!($W, 0f0)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
