using BenchmarkTools
using LinearAlgebra

# Plain `Matrix{T}` + LAPACK approach. Per-thread scratch buffers are
# pre-allocated once per chunk, so the hot loop does no heap allocation.
# This avoids the SMatrix compile-time blowup at D=16 (where the QR of a
# 32×32 SMatrix unrolls into a huge amount of code).

function sqrt_kalman_cpu_mt!(
    Ss_out::Array{T,3}, Ss_in::Array{T,3},
    A::Matrix{T}, S_Q::Matrix{T}, H::Matrix{T}, S_R::Matrix{T},
) where {T}
    D, _, N = size(Ss_in)
    twoD = 2 * D

    nchunks = Threads.nthreads()
    chunk_size = cld(N, nchunks)

    @sync for c in 1:nchunks
        Threads.@spawn begin
            # Per-thread scratch — allocated once per chunk, then reused.
            X        = Matrix{T}(undef, D,    D)
            M_pred   = Matrix{T}(undef, twoD, D)
            Y        = Matrix{T}(undef, D,    D)
            M_upd    = Matrix{T}(undef, twoD, twoD)
            tau_pred = Vector{T}(undef, D)
            tau_upd  = Vector{T}(undef, twoD)

            start_i = (c - 1) * chunk_size + 1
            end_i   = min(c * chunk_size, N)

            @inbounds for i in start_i:end_i
                S_i = view(Ss_in, :, :, i)

                # 1. X = A · S
                mul!(X, A, S_i)

                # 2. Build M_pred = [Xᵀ ; S_Qᵀ]   (2D × D)
                for jj in 1:D, ii in 1:D
                    M_pred[ii,     jj] = X[jj,   ii]
                    M_pred[D + ii, jj] = S_Q[jj, ii]
                end

                # 3. QR(M_pred) via LAPACK geqrf!. Upper triangle of M_pred[1:D, 1:D]
                # holds R_pred; lower triangle holds Householder vectors.
                LAPACK.geqrf!(M_pred, tau_pred)

                # 4. Zero the strict lower triangle of the top-D rows so the next
                # sgemm reads a clean upper-triangular R_pred when transposed.
                for jj in 1:D, ii in (jj + 1):D
                    M_pred[ii, jj] = zero(T)
                end

                # 5. Y = H · R_predᵀ
                R_pred_view = view(M_pred, 1:D, 1:D)
                mul!(Y, H, R_pred_view')

                # 6. Build M_upd = [S_Rᵀ 0 ; Yᵀ R_pred]   (2D × 2D)
                for jj in 1:D, ii in 1:D
                    M_upd[ii,         jj    ] = S_R[jj, ii]                 # top-left
                    M_upd[ii,         D + jj] = zero(T)                     # top-right
                    M_upd[D + ii,     jj    ] = Y[jj, ii]                   # bottom-left
                    M_upd[D + ii,     D + jj] = M_pred[ii, jj]              # bottom-right
                    #   (M_pred[ii, jj] is upper-triangular R_pred since we
                    #    zeroed its strict lower triangle in step 4)
                end

                # 7. QR(M_upd) via LAPACK geqrf!.
                LAPACK.geqrf!(M_upd, tau_upd)

                # 8. S_out = R₂₂ᵀ  (lower triangular, R₂₂ = bottom-right D × D
                # upper-triangle of M_upd after factorisation).
                for jj in 1:D, ii in 1:D
                    Ss_out[ii, jj, i] = (ii >= jj) ?
                        M_upd[D + jj, D + ii] : zero(T)
                end
            end
        end
    end
end

function sqrt_kalman_timing(
    Ss_out, Ss_in, A, S_Q, H, S_R,
    _, ::Val, _, ::Val{:cpu_mt},
)
    N = size(Ss_in, 3)

    bench_results = @benchmark begin
        sqrt_kalman_cpu_mt!($Ss_out, $Ss_in, $A, $S_Q, $H, $S_R)
    end

    return median(bench_results.times) / 1e9 / N
end
