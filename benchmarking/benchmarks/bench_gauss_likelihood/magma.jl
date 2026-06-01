using CUDA
using CUDA: i32
using BenchmarkTools
using LinearAlgebra

# Gaussian log-likelihood with MAGMA:
#
#   p_i = -0.5 · (D · log(2π) + log|Σ_i| + (x_i − μ_i)ᵀ Σ_i⁻¹ (x_i − μ_i))
#
# implemented as:
#   1. δ = x − μ                          (custom kernel)
#   2. cholesky(Σ) → L (lower)            (magma_spotrf_batched, MagmaLower)
#   3. y = L \ δ                          (magmablas_strsm_batched)
#   4. log|Σ| = 2 · Σ log(diag(L))        (custom kernel)
#   5. mahal = |y|²                        (custom kernel)
#   6. p = -0.5 · (D log 2π + log|Σ| + mahal)   (folded into kernel 4 or 5)
#
# Note on SLACK padding: MAGMA's `magmablas_strsm_batched` writes past the
# end of the RHS buffer on the last batch (kernel block size > D for D ≤ 16,
# confirmed by hardware MMU fault under compute-sanitizer in trig_backsolve).
# We over-allocate Σ_storage and δ_storage by SLACK extra slots so the
# last-batch overflow lands in our slack region instead of unmapped memory.

# ─── ccall wrappers ─────────────────────────────────────────────────

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
        uplo, n, dA, lda, info_array, batchCount, queue,
    )
end

function magmablas_strsm_batched!(
    side::Magma.LibMagma.magma_side_t,
    uplo::Magma.LibMagma.magma_uplo_t,
    transA::Magma.LibMagma.magma_trans_t,
    diag::Magma.LibMagma.magma_diag_t,
    m::Integer, n::Integer,
    alpha::Cfloat,
    dA_array, ldda::Integer,
    dB_array, lddb::Integer,
    batchCount::Integer, queue,
)
    ccall(
        (:magmablas_strsm_batched, Magma.LibMagma.libmagma),
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
            Magma.LibMagma.magma_queue_t,
        ),
        side, uplo, transA, diag, m, n, alpha,
        dA_array, ldda, dB_array, lddb,
        batchCount, queue,
    )
    return nothing
end

# ─── Custom kernels ─────────────────────────────────────────────────

# δ = x - μ (elementwise; x, μ, δ all shape (D, N) flat in memory).
function _gl_delta_kernel!(δ, x, μ, total::Int32)
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if idx <= total
        @inbounds δ[idx] = x[idx] - μ[idx]
    end
    return
end

function _gl_delta!(δ::CuArray{Float32, 2}, x::CuArray{Float32, 2}, μ::CuArray{Float32, 2}, D::Int, N::Int)
    nthr = 256
    total = D * N
    nblk = cld(total, nthr)
    @cuda threads=nthr blocks=nblk _gl_delta_kernel!(δ, x, μ, Int32(total))
end

# p[b] = -0.5·(D·log(2π) + 2·Σ log L[i,i,b] + Σ y[i,b]²)
# One thread per batch.
function _gl_finalize_kernel!(p, L, y, D::Int32, N::Int32, log_2pi_D::Float32)
    b = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if b <= N
        log_det_2 = 0f0   # will hold Σ log L[i,i,b]
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

function _gl_finalize!(p::CuArray{Float32, 1}, L::CuArray{Float32, 3}, y::CuArray{Float32, 2},
                      D::Int, N::Int)
    nthr = 256
    nblk = cld(N, nthr)
    log_2pi_D = Float32(D) * log(Float32(2π))
    @cuda threads=nthr blocks=nblk _gl_finalize_kernel!(
        p, L, y, Int32(D), Int32(N), log_2pi_D,
    )
end

# ─── Main gauss-likelihood pipeline ─────────────────────────────────

function gauss_likelihood_magma!(
    dΣ, dδ,
    Σ_storage, δ_storage, x_storage, μ_storage, p_storage,
    info_d, D, N, queue_ptr,
)
    LO = Magma.LibMagma.MagmaLower
    LE = Magma.LibMagma.MagmaLeft
    NT = Magma.LibMagma.MagmaNoTrans
    NUNIT = Magma.LibMagma.MagmaNonUnit

    GC.@preserve Σ_storage δ_storage x_storage μ_storage p_storage begin
        # 1. δ = x - μ
        _gl_delta!(δ_storage, x_storage, μ_storage, D, N)
        CUDA.synchronize()

        # 2. Cholesky: Σ → L (lower triangle), in-place
        magma_spotrf_batched!(
            LO, D, dΣ, D, info_d, N, queue_ptr[],
        )

        # 3. y = L \ δ   (strsm in-place on δ_storage)
        magmablas_strsm_batched!(
            LE, LO, NT, NUNIT,
            D, 1, 1f0,
            dΣ, D, dδ, D,
            N, queue_ptr[],
        )

        Magma.LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)

        # 4-6. Finalize: combine log|Σ|, mahal, constant into p[b]
        _gl_finalize!(p_storage, Σ_storage, δ_storage, D, N)
        CUDA.synchronize()
    end
end

function gauss_likelihood_timing(
    ps_out_cpu, xs_in_cpu, µs_in_cpu, Σs_in_cpu, queue_ptr, _, ::Val{:magma},
)
    D, _, N = size(Σs_in_cpu)
    SLACK = 64   # absorbs MAGMA strsm's per-batch overflow on the last batch

    # Padded Σ buffer (will be factorised in-place per iter)
    Σ_storage = CUDA.zeros(Float32, D, D, N + SLACK)
    Σs_in_d = cu(Σs_in_cpu)
    copyto!(view(Σ_storage, :, :, 1:N), Σs_in_d)

    # Padded δ buffer (gets x-μ, then overwritten by y from strsm)
    δ_storage = CUDA.zeros(Float32, D, N + SLACK)
    x_storage = cu(xs_in_cpu)
    μ_storage = cu(µs_in_cpu)

    p_storage = CUDA.zeros(Float32, N)

    dΣ = CUDA.CUBLAS.unsafe_strided_batch(Σ_storage)
    dδ = CUDA.CUBLAS.unsafe_strided_batch(δ_storage)

    info_d = CUDA.zeros(Magma.LibMagma.magma_int_t, N)

    bench_results = @benchmark begin
        gauss_likelihood_magma!(
            $dΣ, $dδ,
            $Σ_storage, $δ_storage, $x_storage, $μ_storage, $p_storage,
            $info_d, $D, $N, $queue_ptr,
        )
    end setup=begin
        # Restore Σ_storage's first N slots from the original (since spotrf
        # destroys it). δ_storage is fully overwritten by the δ kernel each
        # iter so no reset needed.
        copyto!(view($Σ_storage, :, :, 1:$N), $Σs_in_d)
    end evals=1

    return median(bench_results.times) / 1e9 / N
end
