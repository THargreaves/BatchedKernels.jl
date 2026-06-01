using CUDA
using CUDA: i32
using BenchmarkTools
using LinearAlgebra
using Magma
using Random

# ─── ccall wrappers ─────────────────────────────────────────────────

function roofline_marker_kernel()
    return
end

function magmablas_sgemm_batched!(
    transA, transB, m, n, k,
    alpha, dA, ldda, dB, lddb,
    beta, dC, lddc,
    batchCount, queue,
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
        transA, transB, m, n, k,
        alpha, dA, ldda, dB, lddb,
        beta, dC, lddc,
        batchCount, queue,
    )
end

function magma_sgeqrf_batched!(
    m::Integer, n::Integer,
    dA_array, ldda::Integer,
    dtau_array,
    info_array,
    batchCount::Integer,
    queue::Magma.LibMagma.magma_queue_t,
)
    return ccall(
        (:magma_sgeqrf_batched, Magma.LibMagma.libmagma),
        Magma.LibMagma.magma_int_t,
        (
            Magma.LibMagma.magma_int_t,           # m
            Magma.LibMagma.magma_int_t,           # n
            CuPtr{CuPtr{Cfloat}},                 # dA_array
            Magma.LibMagma.magma_int_t,           # ldda
            CuPtr{CuPtr{Cfloat}},                 # dtau_array
            CuPtr{Magma.LibMagma.magma_int_t},    # info_array
            Magma.LibMagma.magma_int_t,           # batchCount
            Magma.LibMagma.magma_queue_t,
        ),
        m, n, dA_array, ldda, dtau_array,
        info_array, batchCount, queue,
    )
end

# ─── Repeat-pointer helper (one matrix shared across all batch slots) ──

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

# ─── Custom assembly kernels ────────────────────────────────────────
#
# These do the matrix shuffling that no library primitive provides:
# stacking transposes into pre-arrays, zeroing Householder vectors so a
# clean upper-triangular R survives, and extracting/transposing the
# trailing R₂₂ block of the final factorisation.

# M_pred (2D × D × N) = [Xᵀ ; S_Qᵀ] per batch.
# S_Q is the shared lower-triangular sqrt-Q (D × D, broadcast over batches).
function _make_M_pred_kernel!(M_pred, X, S_Q, D::Int32, twoD::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        elem_in_batch = mod(idx - 1i32, twoD * D)
        batch = div(idx - 1i32, twoD * D) + 1i32
        col_zero = div(elem_in_batch, twoD)
        row_zero = mod(elem_in_batch, twoD)
        @inbounds begin
            if row_zero < D
                # Top block: X transposed
                M_pred[row_zero + 1i32, col_zero + 1i32, batch] =
                    X[col_zero + 1i32, row_zero + 1i32, batch]
            else
                # Bottom block: S_Q transposed (shared)
                M_pred[row_zero + 1i32, col_zero + 1i32, batch] =
                    S_Q[col_zero + 1i32, row_zero - D + 1i32]
            end
        end
    end
    return
end

function _make_M_pred!(M_pred::CuArray{Float32,3}, X::CuArray{Float32,3}, S_Q::CuArray{Float32,2}, D::Int, N::Int)
    twoD = 2 * D
    nthreads = 256
    total = twoD * D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _make_M_pred_kernel!(
        M_pred, X, S_Q, Int32(D), Int32(twoD), Int32(total),
    )
end

# Zero out the strict lower triangle of the top-D rows of M_pred (which holds
# Householder vectors after geqrf). After this, M_pred[1:D, 1:D] is exactly
# R_pred (upper triangular, zeros below) and can be passed to sgemm/transB
# without corruption.
function _zero_lower_triangle_top_kernel!(M_pred, D::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        elem_in_batch = mod(idx - 1i32, D * D)
        batch = div(idx - 1i32, D * D) + 1i32
        col_zero = div(elem_in_batch, D)
        row_zero = mod(elem_in_batch, D)
        if row_zero > col_zero
            @inbounds M_pred[row_zero + 1i32, col_zero + 1i32, batch] = 0f0
        end
    end
    return
end

function _zero_lower_triangle_top!(M_pred::CuArray{Float32,3}, D::Int, N::Int)
    nthreads = 256
    total = D * D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _zero_lower_triangle_top_kernel!(
        M_pred, Int32(D), Int32(total),
    )
end

# M_upd (2D × 2D × N):
#   [S_Rᵀ   0      ]
#   [Yᵀ     R_pred ]
# S_R shared lower-triangular sqrt-R; Y batched (D×D); R_pred batched (upper
# triangular, lives in M_pred[1:D, 1:D] with leading dim 2D).
function _make_M_upd_kernel!(M_upd, S_R, Y, M_pred, D::Int32, twoD::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        elem_in_batch = mod(idx - 1i32, twoD * twoD)
        batch = div(idx - 1i32, twoD * twoD) + 1i32
        col_zero = div(elem_in_batch, twoD)
        row_zero = mod(elem_in_batch, twoD)
        @inbounds begin
            if row_zero < D && col_zero < D
                # Top-left: S_R transposed (shared)
                M_upd[row_zero + 1i32, col_zero + 1i32, batch] =
                    S_R[col_zero + 1i32, row_zero + 1i32]
            elseif row_zero < D
                # Top-right: zero
                M_upd[row_zero + 1i32, col_zero + 1i32, batch] = 0f0
            elseif col_zero < D
                # Bottom-left: Y transposed
                M_upd[row_zero + 1i32, col_zero + 1i32, batch] =
                    Y[col_zero + 1i32, row_zero - D + 1i32, batch]
            else
                # Bottom-right: R_pred upper triangle (lower zero)
                r_row = row_zero - D
                r_col = col_zero - D
                if r_row <= r_col
                    M_upd[row_zero + 1i32, col_zero + 1i32, batch] =
                        M_pred[r_row + 1i32, r_col + 1i32, batch]
                else
                    M_upd[row_zero + 1i32, col_zero + 1i32, batch] = 0f0
                end
            end
        end
    end
    return
end

function _make_M_upd!(M_upd::CuArray{Float32,3}, S_R::CuArray{Float32,2},
                     Y::CuArray{Float32,3}, M_pred::CuArray{Float32,3},
                     D::Int, N::Int)
    twoD = 2 * D
    nthreads = 256
    total = twoD * twoD * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _make_M_upd_kernel!(
        M_upd, S_R, Y, M_pred, Int32(D), Int32(twoD), Int32(total),
    )
end

# Extract S_out = R₂₂ᵀ (lower triangular, D × D × N) from the factored M_upd
# (which holds upper-triangular R in its upper triangle after geqrf).
# R₂₂ is the bottom-right D × D block of R, at M_upd[D+1:2D, D+1:2D].
function _extract_R22_transposed_kernel!(S_out, M_upd, D::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        elem_in_batch = mod(idx - 1i32, D * D)
        batch = div(idx - 1i32, D * D) + 1i32
        col_zero = div(elem_in_batch, D)
        row_zero = mod(elem_in_batch, D)
        @inbounds begin
            if row_zero >= col_zero
                # Lower triangle (incl. diagonal): S_out[row, col] = R₂₂[col, row]
                S_out[row_zero + 1i32, col_zero + 1i32, batch] =
                    M_upd[D + col_zero + 1i32, D + row_zero + 1i32, batch]
            else
                # Upper triangle: zero (S_out is the lower-triangular sqrt)
                S_out[row_zero + 1i32, col_zero + 1i32, batch] = 0f0
            end
        end
    end
    return
end

function _extract_R22_transposed!(S_out::CuArray{Float32,3}, M_upd::CuArray{Float32,3},
                                  D::Int, N::Int)
    nthreads = 256
    total = D * D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _extract_R22_transposed_kernel!(
        S_out, M_upd, Int32(D), Int32(total),
    )
end

# ─── Main square-root Kalman update ─────────────────────────────────

function sqrt_kalman_magma!(
    # Pointer arrays
    dSs_in, dX, dM_pred, dY, dM_upd, dSs_out,
    dtau_pred, dtau_upd,
    dA_repeat, dH_repeat,
    # Underlying buffers (kept alive via GC.@preserve + function args)
    Ss_in, X, M_pred, Y, M_upd, Ss_out,
    tau_pred, tau_upd,
    A, H, S_Q, S_R,
    info_pred, info_upd,
    D, N, queue_ptr,
)
    NT = Magma.LibMagma.MagmaNoTrans
    TR = Magma.LibMagma.MagmaTrans
    twoD = 2 * D

    GC.@preserve Ss_in X M_pred Y M_upd Ss_out tau_pred tau_upd A H S_Q S_R begin
        # 1. X := A · S_in        (sgemm, shared A on left, batched S on right)
        magmablas_sgemm_batched!(
            NT, NT, D, D, D,
            1f0, dA_repeat, D, dSs_in, D,
            0f0, dX, D, N, queue_ptr[],
        )

        # 2. M_pred := [Xᵀ; S_Qᵀ]   (custom kernel on default stream)
        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        _make_M_pred!(M_pred, X, S_Q, D, N)
        CUDA.synchronize()

        # 3. QR(M_pred), 2D × D.  R_pred lives in upper-triangle of M_pred[1:D, 1:D].
        magma_sgeqrf_batched!(
            twoD, D, dM_pred, twoD, dtau_pred, info_pred, N, queue_ptr[],
        )

        # 4. Zero out the strict lower triangle of M_pred[1:D, 1:D]
        # so step 5's sgemm with transB reads clean R_pred.
        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        _zero_lower_triangle_top!(M_pred, D, N)
        CUDA.synchronize()

        # 5. Y := H · R_predᵀ      (sgemm, shared H on left, batched R_pred on right
        #                            with leading dim 2D)
        magmablas_sgemm_batched!(
            NT, TR, D, D, D,
            1f0, dH_repeat, D, dM_pred, twoD,
            0f0, dY, D, N, queue_ptr[],
        )

        # 6. M_upd := [S_Rᵀ 0; Yᵀ R_pred]   (custom kernel on default stream)
        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        _make_M_upd!(M_upd, S_R, Y, M_pred, D, N)
        CUDA.synchronize()

        # 7. QR(M_upd), 2D × 2D.
        magma_sgeqrf_batched!(
            twoD, twoD, dM_upd, twoD, dtau_upd, info_upd, N, queue_ptr[],
        )

        # 8. S_out := R₂₂ᵀ          (lower-triangular sqrt of new P)
        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
        _extract_R22_transposed!(Ss_out, M_upd, D, N)
        CUDA.synchronize()
    end
end

function launch_magma(Ss_out_cpu, Ss_in_cpu, A_cpu, S_Q_cpu, H_cpu, S_R_cpu, queue_ptr, loop_count)
    D, _, N = size(Ss_out_cpu)
    twoD = 2 * D

    # Move shared and per-batch inputs to GPU.
    Ss_in = cu(Ss_in_cpu)
    A     = cu(A_cpu)
    S_Q   = cu(S_Q_cpu)
    H     = cu(H_cpu)
    S_R   = cu(S_R_cpu)

    # Mutable per-batch storages (overwritten by the pipeline each iter).
    X       = CUDA.zeros(Float32, D,    D,    N)
    M_pred  = CUDA.zeros(Float32, twoD, D,    N)
    Y       = CUDA.zeros(Float32, D,    D,    N)
    M_upd   = CUDA.zeros(Float32, twoD, twoD, N)
    Ss_out  = cu(Ss_out_cpu)

    # geqrf τ buffers.
    tau_pred = CUDA.zeros(Float32, D,    N)
    tau_upd  = CUDA.zeros(Float32, twoD, N)

    # Pointer arrays (host-built once, valid for the whole benchmark).
    dSs_in     = CUDA.CUBLAS.unsafe_strided_batch(Ss_in)
    dX         = CUDA.CUBLAS.unsafe_strided_batch(X)
    dM_pred    = CUDA.CUBLAS.unsafe_strided_batch(M_pred)
    dY         = CUDA.CUBLAS.unsafe_strided_batch(Y)
    dM_upd     = CUDA.CUBLAS.unsafe_strided_batch(M_upd)
    dSs_out    = CUDA.CUBLAS.unsafe_strided_batch(Ss_out)
    dtau_pred  = CUDA.CUBLAS.unsafe_strided_batch(tau_pred)
    dtau_upd   = CUDA.CUBLAS.unsafe_strided_batch(tau_upd)
    dA_repeat  = unsafe_strided_batch_repeat(A, N)
    dH_repeat  = unsafe_strided_batch_repeat(H, N)

    info_pred = CUDA.zeros(Magma.LibMagma.magma_int_t, N)
    info_upd  = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    for i in 1:loop_count
        if i == loop_count
            CUDA.@sync @cuda threads = 1 blocks = 1 roofline_marker_kernel()
            CUDA.synchronize()
        end
        sqrt_kalman_magma!(
            dSs_in, dX, dM_pred, dY, dM_upd, dSs_out,
            dtau_pred, dtau_upd,
            dA_repeat, dH_repeat,
            Ss_in, X, M_pred, Y, M_upd, Ss_out,
            tau_pred, tau_upd,
            A, H, S_Q, S_R,
            info_pred, info_upd,
            D, N, queue_ptr,
        )
    end
end

function main(D, loop_count)
    Random.seed!(1234)
    N = Int(ceil(1e9 / (4 * 2 * D^2)))
    T = Float32

    S_in = Array{Float32}(undef, D, D, N)
    for i in 1:N
        P_i = rand(Float32, D, D) / Float32(D)
        P_i = P_i * P_i' + 0.1f0 * I
        S_in[:, :, i] = Float32.(Matrix(cholesky(P_i).L))
    end
    S_out = zeros(Float32, D, D, N)

    A = rand(T, D, D) / Float32(D)

    Q_elem = rand(Float32, D, D) / Float32(D)^2
    Q_elem = Q_elem * Q_elem' + 0.01f0 * I
    S_Q = Float32.(Matrix(cholesky(Q_elem).L))

    H = rand(Float32, D, D) / Float32(D)

    R_elem = rand(Float32, D, D) / Float32(D)^2
    R_elem = R_elem * R_elem' + 0.01f0 * I
    S_R = Float32.(Matrix(cholesky(R_elem).L))

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

    launch_magma(S_out, S_in, A, S_Q, H, S_R, queue_ptr, loop_count)
end

D = parse(Int, ARGS[1])
loop_count = parse(Int, ARGS[2])
main(D, loop_count)
