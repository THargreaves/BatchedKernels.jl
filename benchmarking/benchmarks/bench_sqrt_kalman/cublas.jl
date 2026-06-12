# cublas.jl
#
# cuBLAS implementation of the square-root Kalman update.
# Uses:
#   - cublasSgemmStridedBatched for the two batched matmuls (with strideA = 0
#     to broadcast the shared A / H matrices across all batch slots)
#   - cublasSgeqrfBatched for both QR factorisations (rectangular 2D × D and
#     square 2D × 2D — neither qualifies for cuSOLVER's smallsq variant)
#   - Custom CUDA kernels for the pre-array assembly and final extraction
#
# Kernels are prefixed `_cublas_*` to avoid colliding with magma.jl when both
# are include()d in the same session (e.g. by run_script.jl).

using CUDA
using CUDA: i32
using BenchmarkTools
using Statistics
using LinearAlgebra

# ─── Device pointer-array builder (for QR; sgemm uses stride API) ──────

function _cublas_fill_batch_ptrs_kernel!(ptrs, A, slice_elems)
    tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    gstride = gridDim().x * blockDim().x
    n = length(ptrs)
    i = tid
    while i <= n
        idx = (Int64(i) - 1) * Int64(slice_elems) + 1
        @inbounds ptrs[i] = reinterpret(Ptr{Float32}, pointer(A, idx))
        i += gstride
    end
    return
end

function _cublas_batch_ptrs(A::DenseCuArray{Float32}, N::Integer, slice_elems::Integer)
    ptrs = CuArray{Ptr{Float32}}(undef, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _cublas_fill_batch_ptrs_kernel!(
        ptrs, A, Int32(slice_elems),
    )
    return ptrs
end

# ─── Assembly kernels (same shape as MAGMA's; prefixed to avoid clash) ─

function _cublas_make_M_pred_kernel!(M_pred, X, S_Q, D::Int32, twoD::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        elem_in_batch = mod(idx - 1i32, twoD * D)
        batch = div(idx - 1i32, twoD * D) + 1i32
        col_zero = div(elem_in_batch, twoD)
        row_zero = mod(elem_in_batch, twoD)
        @inbounds begin
            if row_zero < D
                M_pred[row_zero + 1i32, col_zero + 1i32, batch] =
                    X[col_zero + 1i32, row_zero + 1i32, batch]
            else
                M_pred[row_zero + 1i32, col_zero + 1i32, batch] =
                    S_Q[col_zero + 1i32, row_zero - D + 1i32]
            end
        end
    end
    return
end

function _cublas_make_M_pred!(M_pred, X, S_Q, D::Int, N::Int)
    twoD = 2 * D
    nthreads = 256
    total = twoD * D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _cublas_make_M_pred_kernel!(
        M_pred, X, S_Q, Int32(D), Int32(twoD), Int32(total),
    )
end

function _cublas_zero_lower_triangle_top_kernel!(M_pred, D::Int32, total::Int32)
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

function _cublas_zero_lower_triangle_top!(M_pred, D::Int, N::Int)
    nthreads = 256
    total = D * D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _cublas_zero_lower_triangle_top_kernel!(
        M_pred, Int32(D), Int32(total),
    )
end

function _cublas_make_M_upd_kernel!(M_upd, S_R, Y, M_pred, D::Int32, twoD::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        elem_in_batch = mod(idx - 1i32, twoD * twoD)
        batch = div(idx - 1i32, twoD * twoD) + 1i32
        col_zero = div(elem_in_batch, twoD)
        row_zero = mod(elem_in_batch, twoD)
        @inbounds begin
            if row_zero < D && col_zero < D
                M_upd[row_zero + 1i32, col_zero + 1i32, batch] =
                    S_R[col_zero + 1i32, row_zero + 1i32]
            elseif row_zero < D
                M_upd[row_zero + 1i32, col_zero + 1i32, batch] = 0f0
            elseif col_zero < D
                M_upd[row_zero + 1i32, col_zero + 1i32, batch] =
                    Y[col_zero + 1i32, row_zero - D + 1i32, batch]
            else
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

function _cublas_make_M_upd!(M_upd, S_R, Y, M_pred, D::Int, N::Int)
    twoD = 2 * D
    nthreads = 256
    total = twoD * twoD * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _cublas_make_M_upd_kernel!(
        M_upd, S_R, Y, M_pred, Int32(D), Int32(twoD), Int32(total),
    )
end

function _cublas_extract_R22_transposed_kernel!(S_out, M_upd, D::Int32, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        elem_in_batch = mod(idx - 1i32, D * D)
        batch = div(idx - 1i32, D * D) + 1i32
        col_zero = div(elem_in_batch, D)
        row_zero = mod(elem_in_batch, D)
        @inbounds begin
            if row_zero >= col_zero
                S_out[row_zero + 1i32, col_zero + 1i32, batch] =
                    M_upd[D + col_zero + 1i32, D + row_zero + 1i32, batch]
            else
                S_out[row_zero + 1i32, col_zero + 1i32, batch] = 0f0
            end
        end
    end
    return
end

function _cublas_extract_R22_transposed!(S_out, M_upd, D::Int, N::Int)
    nthreads = 256
    total = D * D * N
    nblocks = cld(total, nthreads)
    @cuda threads=nthreads blocks=nblocks _cublas_extract_R22_transposed_kernel!(
        S_out, M_upd, Int32(D), Int32(total),
    )
end

# ─── Main square-root Kalman update ─────────────────────────────────

function sqrt_kalman_cublas!(
    dM_pred_ptrs, dM_upd_ptrs,
    dtau_pred_ptrs, dtau_upd_ptrs,
    Ss_in, X, M_pred, Y, M_upd, Ss_out,
    tau_pred, tau_upd,
    A, H, S_Q, S_R,
    info_pred, info_upd,
    D, N,
)
    h = CUDA.CUBLAS.handle()
    OP_N = CUDA.CUBLAS.CUBLAS_OP_N
    OP_T = CUDA.CUBLAS.CUBLAS_OP_T
    twoD = 2 * D

    GC.@preserve Ss_in X M_pred Y M_upd Ss_out tau_pred tau_upd A H S_Q S_R begin
        # 1. X := A · S_in   (strided-batched sgemm; strideA = 0 broadcasts shared A)
        CUDA.CUBLAS.cublasSgemmStridedBatched(
            h, OP_N, OP_N, D, D, D,
            1f0,
            A, D, 0,                    # shared A: stride 0
            Ss_in, D, D * D,            # batched S_in
            0f0,
            X, D, D * D,                # batched X
            N,
        )

        # 2. M_pred := [Xᵀ; S_Qᵀ]
        _cublas_make_M_pred!(M_pred, X, S_Q, D, N)

        # 3. QR(M_pred), 2D × D — pointer-array geqrf (no strided variant exists)
        info_h = Ref{Cint}(0)
        CUDA.CUBLAS.cublasSgeqrfBatched(
            h, twoD, D, dM_pred_ptrs, twoD, dtau_pred_ptrs, info_h, N,
        )

        # 4. Zero Householder vectors below R_pred's diagonal
        _cublas_zero_lower_triangle_top!(M_pred, D, N)

        # 5. Y := H · R_predᵀ   (sgemm with transB; M_pred has leading dim 2D)
        CUDA.CUBLAS.cublasSgemmStridedBatched(
            h, OP_N, OP_T, D, D, D,
            1f0,
            H, D, 0,                    # shared H: stride 0
            M_pred, twoD, twoD * D,     # batched M_pred (top D rows = R_pred)
            0f0,
            Y, D, D * D,                # batched Y
            N,
        )

        # 6. M_upd := [S_Rᵀ 0; Yᵀ R_pred]
        _cublas_make_M_upd!(M_upd, S_R, Y, M_pred, D, N)

        # 7. QR(M_upd), 2D × 2D
        info_h2 = Ref{Cint}(0)
        CUDA.CUBLAS.cublasSgeqrfBatched(
            h, twoD, twoD, dM_upd_ptrs, twoD, dtau_upd_ptrs, info_h2, N,
        )

        # 8. S_out := R₂₂ᵀ
        _cublas_extract_R22_transposed!(Ss_out, M_upd, D, N)

        CUDA.synchronize()
    end
end

function sqrt_kalman_timing(
    Ss_out_cpu, Ss_in_cpu, A_cpu, S_Q_cpu, H_cpu, S_R_cpu,
    _, ::Val{THRESH}, _, ::Val{:cublas},
) where {THRESH}
    D, _, N = size(Ss_out_cpu)
    twoD = 2 * D

    Ss_in = cu(Ss_in_cpu)
    A     = cu(A_cpu)
    S_Q   = cu(S_Q_cpu)
    H     = cu(H_cpu)
    S_R   = cu(S_R_cpu)

    X       = CUDA.zeros(Float32, D,    D,    N)
    M_pred  = CUDA.zeros(Float32, twoD, D,    N)
    Y       = CUDA.zeros(Float32, D,    D,    N)
    M_upd   = CUDA.zeros(Float32, twoD, twoD, N)
    Ss_out  = cu(Ss_out_cpu)

    tau_pred = CUDA.zeros(Float32, D,    N)
    tau_upd  = CUDA.zeros(Float32, twoD, N)

    # Pointer arrays (only needed for the QR calls, not sgemm)
    dM_pred_ptrs  = _cublas_batch_ptrs(M_pred,  N, twoD * D)
    dM_upd_ptrs   = _cublas_batch_ptrs(M_upd,   N, twoD * twoD)
    dtau_pred_ptrs = _cublas_batch_ptrs(tau_pred, N, D)
    dtau_upd_ptrs  = _cublas_batch_ptrs(tau_upd,  N, twoD)

    info_pred = Ref{Cint}(0)
    info_upd  = Ref{Cint}(0)

    bench_results = @benchmark begin
        sqrt_kalman_cublas!(
            $dM_pred_ptrs, $dM_upd_ptrs,
            $dtau_pred_ptrs, $dtau_upd_ptrs,
            $Ss_in, $X, $M_pred, $Y, $M_upd, $Ss_out,
            $tau_pred, $tau_upd,
            $A, $H, $S_Q, $S_R,
            $info_pred, $info_upd,
            $D, $N,
        )
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
