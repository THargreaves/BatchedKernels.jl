# cublas_cusolver.jl
#
# Gaussian log-likelihood with cuBLAS + cuSOLVER.
#   - cusolverDnSpotrfBatched : Cholesky factorisation (lower)
#   - cublasStrsmBatched      : triangular solve L · y = δ
#   - custom CUDA kernels     : δ = x − μ, log|Σ|, mahal, finalise
#
# Both libraries dispatch on the current CUDA stream (same as the kalman
# cublas_cusolver), so no MAGMA queue is needed. Everything is implicitly
# serialised — just one final CUDA.synchronize() to capture completion.

using CUDA
using CUDA: i32
using BenchmarkTools
using Statistics
using LinearAlgebra

# ─── Device pointer-array builder ───────────────────────────────────

function _fill_batch_ptrs_kernel!(ptrs, A, slice_elems)
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

function _batch_ptrs(A::DenseCuArray{Float32}, N::Integer, slice_elems::Integer)
    ptrs = CuArray{Ptr{Float32}}(undef, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _fill_batch_ptrs_kernel!(
        ptrs, A, Int32(slice_elems),
    )
    return ptrs
end

# ─── Custom kernels (prefixed `_cc_*` to avoid clash with magma.jl) ─

function _cc_gl_delta_kernel!(δ, x, μ, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        @inbounds δ[idx] = x[idx] - μ[idx]
    end
    return
end

function _cc_gl_delta!(δ::CuArray{Float32, 2}, x::CuArray{Float32, 2}, μ::CuArray{Float32, 2}, D::Int, N::Int)
    nthr = 256
    total = D * N
    nblk = cld(total, nthr)
    @cuda threads=nthr blocks=nblk _cc_gl_delta_kernel!(δ, x, μ, Int32(total))
end

function _cc_gl_finalize_kernel!(p, L, y, D::Int32, N::Int32, log_2pi_D::Float32)
    b = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if b <= N
        log_det_2 = 0f0
        mahal = 0f0
        @inbounds for i in 1i32:D
            log_det_2 += log(L[i, i, b])
            yi = y[i, b]
            mahal += yi * yi
        end
        @inbounds p[b] = -0.5f0 * (log_2pi_D + 2f0 * log_det_2 + mahal)
    end
    return
end

function _cc_gl_finalize!(p::CuArray{Float32, 1}, L::CuArray{Float32, 3}, y::CuArray{Float32, 2},
                          D::Int, N::Int)
    nthr = 256
    nblk = cld(N, nthr)
    log_2pi_D = Float32(D) * log(Float32(2π))
    @cuda threads=nthr blocks=nblk _cc_gl_finalize_kernel!(
        p, L, y, Int32(D), Int32(N), log_2pi_D,
    )
end

# ─── Main pipeline ──────────────────────────────────────────────────

function gauss_likelihood_cublas_cusolver!(
    dΣ, dδ,
    Σ_storage, δ_storage, x_storage, μ_storage, p_storage,
    info_d, D, N,
)
    h  = CUDA.CUBLAS.handle()
    sh = CUDA.CUSOLVER.dense_handle()

    LEFT    = CUDA.CUBLAS.CUBLAS_SIDE_LEFT
    LOWER   = CUDA.CUBLAS.CUBLAS_FILL_MODE_LOWER
    OP_N    = CUDA.CUBLAS.CUBLAS_OP_N
    NONUNIT = CUDA.CUBLAS.CUBLAS_DIAG_NON_UNIT

    GC.@preserve Σ_storage δ_storage x_storage μ_storage p_storage begin
        # 1. δ = x − μ
        _cc_gl_delta!(δ_storage, x_storage, μ_storage, D, N)

        # 2. Cholesky: Σ → L (lower), in-place
        CUDA.CUSOLVER.cusolverDnSpotrfBatched(sh, LOWER, D, dΣ, D, info_d, N)

        # 3. y = L \ δ   (strsm in-place on δ_storage)
        CUDA.CUBLAS.cublasStrsmBatched(
            h, LEFT, LOWER, OP_N, NONUNIT,
            D, 1, 1f0, dΣ, D, dδ, D, N,
        )

        # 4-6. log|Σ|, mahal, and constant folded into per-batch p
        _cc_gl_finalize!(p_storage, Σ_storage, δ_storage, D, N)

        CUDA.synchronize()
    end
end

function gauss_likelihood_timing(
    ps_out_cpu, xs_in_cpu, µs_in_cpu, Σs_in_cpu, _, _, ::Val{:cublas_cusolver},
)
    D, _, N = size(Σs_in_cpu)
    DD = D * D

    # Σ buffer (factorised in-place per iter, restored from Σs_in_d in setup)
    Σ_storage = cu(Σs_in_cpu)
    Σs_in_d   = cu(Σs_in_cpu)

    # δ buffer (gets x-μ, then overwritten by y from strsm)
    δ_storage = CUDA.zeros(Float32, D, N)
    x_storage = cu(xs_in_cpu)
    μ_storage = cu(µs_in_cpu)

    p_storage = CUDA.zeros(Float32, N)

    dΣ = _batch_ptrs(Σ_storage, N, DD)
    dδ = _batch_ptrs(δ_storage, N, D)

    info_d = CUDA.zeros(Cint, N)

    bench_results = @benchmark begin
        gauss_likelihood_cublas_cusolver!(
            $dΣ, $dδ,
            $Σ_storage, $δ_storage, $x_storage, $μ_storage, $p_storage,
            $info_d, $D, $N,
        )
    end setup=begin
        # Restore Σ_storage from the original (spotrf destroys it).
        copyto!($Σ_storage, $Σs_in_d)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
