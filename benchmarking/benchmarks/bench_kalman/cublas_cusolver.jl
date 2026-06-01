# cublas.jl
#
# cuBLAS / cuSOLVER batched equivalent of the MAGMA Kalman covariance update.
# Drop-in for the benchmark harness — same `kalman_timing(...)` API.
#
# Usage:
#   1. `include("cublas.jl")`  (after `include("magma.jl")` is fine, but this
#       file is self-contained and has no load-order dependency).
#   2. Add `Val(:cublas) => "cuBLAS"` to the `methods` dict in the caller.
#
# Implementation notes:
#   - Mirrors `kalman_magma!` step-for-step (same 12-step covariance update).
#   - GEMMs use `cublasSgemmBatched` (pointer-array batched API).
#   - cuBLAS has no batched Cholesky, so step 7 uses `cusolverDnSpotrfBatched`.
#   - Triangular solves use `cublasStrsmBatched`.
#   - Shared matrices (A, H) are broadcast via *repeated device pointers*, just
#     like `unsafe_strided_batch_repeat` in the MAGMA version — every batch slot
#     points at the same D×D matrix, so there is no extra memory/bandwidth cost.
#   - cuBLAS, cuSOLVER and the custom `@cuda` kernels all run on the current
#     CUDA stream, so everything is implicitly serialized. Unlike the MAGMA
#     version (separate queue) NO inter-step syncs are required — just one
#     final `CUDA.synchronize()` so the benchmark captures completion.

using CUDA
using CUDA: i32
using BenchmarkTools
using Statistics

# ─── Device pointer-array builders ──────────────────────────────────

# ptrs[i] → start of batch slice i within a D×D×N contiguous array.
function _fill_batch_ptrs_kernel!(ptrs, A, DD)
    tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    gstride = gridDim().x * blockDim().x
    n = length(ptrs)
    i = tid
    while i <= n
        idx = (Int64(i) - 1) * Int64(DD) + 1          # Int64 to avoid overflow
        @inbounds ptrs[i] = reinterpret(Ptr{Float32}, pointer(A, idx))
        i += gstride
    end
    return
end

# ptrs[i] → same matrix for every i  (shared / broadcast operand).
function _fill_repeat_ptrs_kernel!(ptrs, A)
    tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    gstride = gridDim().x * blockDim().x
    n = length(ptrs)
    p = reinterpret(Ptr{Float32}, pointer(A, 1))
    i = tid
    while i <= n
        @inbounds ptrs[i] = p
        i += gstride
    end
    return
end

function _batch_ptrs(A::DenseCuArray{Float32}, N::Integer, DD::Integer)
    ptrs = CuArray{Ptr{Float32}}(undef, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _fill_batch_ptrs_kernel!(ptrs, A, Int32(DD))
    return ptrs
end

function _repeat_ptrs(A::DenseCuArray{Float32}, N::Integer)
    ptrs = CuArray{Ptr{Float32}}(undef, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _fill_repeat_ptrs_kernel!(ptrs, A)
    return ptrs
end

# ─── Elementwise helper kernels (shared-matrix add, batched sub) ─────

function _cublas_add_shared_kernel!(dst, src, DD, total)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= total
        @inbounds dst[i] += src[((i - 1i32) % DD) + 1i32]
    end
    return
end

function _cublas_sub_batched_kernel!(dst, src, total)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= total
        @inbounds dst[i] -= src[i]
    end
    return
end

# dst[i] += src[((i-1) % DD) + 1]: adds a shared D×D matrix into each slot
function _cublas_add_shared!(dst, src, DD::Int, N::Int)
    total = DD * N
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _cublas_add_shared_kernel!(
        dst, src, Int32(DD), Int32(total))
end

# dst[i] -= src[i]
function _cublas_sub_batched!(dst, src, total::Int)
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _cublas_sub_batched_kernel!(
        dst, src, Int32(total))
end

# ─── Main Kalman covariance update ──────────────────────────────────
#
# Slot usage (identical to kalman_magma!):
#   P_in           P_out           W
#  ───────────────────────────────────────────
#  1.  P_in          -            A*P_in
#  2.  -             A*P*A'       -
#  3.  -             P_pred       -
#  4.  P_pred*H'     P_pred       -
#  5.  P_pred*H'     P_pred       H*P_pred*H'
#  6.  P_pred*H'     P_pred       S
#  7.  P_pred*H'     P_pred       L
#  8.  P_pred*H'L⁻ᵀ  P_pred       L
#  9.  K             P_pred       L
# 10.  K             P_pred       K*H
# 11.  K*H*P_pred    P_pred       -
# 12.  -            (I-K*H)*P     -

function kalman_cublas!(
    dPo, dPi, dW,        # batched pointer arrays   (CuVector{Ptr{Float32}})
    dA, dH,              # shared (repeated) pointer arrays
    P_out, P_in, W,      # flat CuArrays for the custom kernels
    Q, R,                # shared D*D flat CuArrays (read-only)
    info_d, D, N,
)
    h  = CUDA.CUBLAS.handle()           # cuBLAS handle, bound to current stream
    sh = CUDA.CUSOLVER.dense_handle()   # cuSOLVER dense handle, current stream

    OP_N    = CUDA.CUBLAS.CUBLAS_OP_N
    OP_T    = CUDA.CUBLAS.CUBLAS_OP_T
    LOWER   = CUDA.CUBLAS.CUBLAS_FILL_MODE_LOWER
    RIGHT   = CUDA.CUBLAS.CUBLAS_SIDE_RIGHT
    NONUNIT = CUDA.CUBLAS.CUBLAS_DIAG_NON_UNIT

    α = 1.0f0
    β = 0.0f0
    DD = D * D

    # 1. W = A * P_in
    CUDA.CUBLAS.cublasSgemmBatched(h, OP_N, OP_N, D, D, D,
        α, dA, D, dPi, D, β, dW, D, N)

    # 2. P_out = W * A'            (= A P A')
    CUDA.CUBLAS.cublasSgemmBatched(h, OP_N, OP_T, D, D, D,
        α, dW, D, dA, D, β, dPo, D, N)

    # 3. P_out += Q                (= P_pred)
    _cublas_add_shared!(P_out, Q, DD, N)

    # 4. P_in = P_pred * H'
    CUDA.CUBLAS.cublasSgemmBatched(h, OP_N, OP_T, D, D, D,
        α, dPo, D, dH, D, β, dPi, D, N)

    # 5. W = H * (P_pred*H')       (= H P_pred H')
    CUDA.CUBLAS.cublasSgemmBatched(h, OP_N, OP_N, D, D, D,
        α, dH, D, dPi, D, β, dW, D, N)

    # 6. W += R                    (= S)
    _cublas_add_shared!(W, R, DD, N)

    # 7. Cholesky: S = L*L'        (L overwrites W, lower triangle)
    CUDA.CUSOLVER.cusolverDnSpotrfBatched(sh, LOWER, D, dW, D, info_d, N)

    # 8. trsm: P_in = P_in * L^{-T}   (P_in holds P_pred*H')
    CUDA.CUBLAS.cublasStrsmBatched(h, RIGHT, LOWER, OP_T, NONUNIT,
        D, D, α, dW, D, dPi, D, N)

    # 9. trsm: P_in = P_in * L^{-1}   (= K = P_pred H' S^{-1})
    CUDA.CUBLAS.cublasStrsmBatched(h, RIGHT, LOWER, OP_N, NONUNIT,
        D, D, α, dW, D, dPi, D, N)

    # 10. W = K * H
    CUDA.CUBLAS.cublasSgemmBatched(h, OP_N, OP_N, D, D, D,
        α, dPi, D, dH, D, β, dW, D, N)

    # 11. P_in = (K*H) * P_pred    (= W * P_out)
    CUDA.CUBLAS.cublasSgemmBatched(h, OP_N, OP_N, D, D, D,
        α, dW, D, dPo, D, β, dPi, D, N)

    # 12. P_out -= P_in            (= P_pred - K*H*P_pred = (I-K*H)*P_pred)
    _cublas_sub_batched!(P_out, P_in, DD * N)

    CUDA.synchronize()
    return nothing
end

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, _, _, ::Val{:cublas_cusolver})
    D, _, N = size(P_in_cpu)
    DD = D * D

    P_out = cu(P_out_cpu)
    P_in  = cu(P_in_cpu)
    A     = cu(A_cpu)
    Q     = cu(Q_cpu)
    H     = cu(H_cpu)
    R     = cu(R_cpu)
    W     = CUDA.zeros(Float32, D, D, N)

    # Pointer arrays are built once. `copyto!` / `fill!` in the benchmark setup
    # reuse the same device buffers, so these pointers stay valid throughout.
    dPo = _batch_ptrs(P_out, N, DD)
    dPi = _batch_ptrs(P_in,  N, DD)
    dW  = _batch_ptrs(W,     N, DD)
    dA  = _repeat_ptrs(A, N)
    dH  = _repeat_ptrs(H, N)

    info_d = CUDA.zeros(Cint, N)

    # Flat views into contiguous GPU memory for the custom kernels
    P_out_flat = reshape(P_out, :)
    P_in_flat  = reshape(P_in, :)
    W_flat     = reshape(W, :)
    Q_flat     = reshape(Q, :)
    R_flat     = reshape(R, :)

    bench_results = @benchmark begin
        kalman_cublas!(
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
        )
    end setup=begin
        copyto!($P_out, $P_out_cpu)
        copyto!($P_in, $P_in_cpu)
        fill!($W, 0f0)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
