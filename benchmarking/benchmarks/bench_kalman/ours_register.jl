using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32
using LinearAlgebra
using Random: MersenneTwister
using StaticArrays: MVector

# =============================================================================
# Register-resident Kalman covariance update (hardcoded, Float32).
#
# Contrast with `ours.jl`: instead of staging every intermediate matrix through
# the dual-access shared layout (shmem_1/2/3), each lane keeps *one column* of
# the working matrices in registers. Only the batch-shared matrices A, Q, H, R
# live in shared memory. Cross-lane access (matmul operand gather, Cholesky
# pivot, triangular factor reads, transpose) is realised with warp shuffles.
#
# Lane `d = mod1(lid, D)` of matrix `warp_matrix_id` owns column `d`. The three
# register slots r1/r2/r3 replace shmem_1/2/3 one-for-one.
# =============================================================================

# out[i] = Σ_k A[i,k]·x[k]   (A shared D×D, x = right operand's column d).
# Left operand is shared → readable by every lane, so NO shuffles.
@inline function _shared_mul_col!(out, A, x, ::Val{D}) where {D}
    @inbounds for i in 1i32:D
        s = 0.0f0
        for k in 1i32:D
            s += A[i, k] * x[k]
        end
        out[i] = s
    end
    return nothing
end

# out = (M · N')[:, d]   with M register col-per-lane, N shared.
# (M·N')[r,d] = Σ_k M[r,k]·N[d,k];  M[r,k] is gathered by broadcasting lane
# (base+k)'s column. D² shuffles.
@inline function _regmul_sharedT!(out, M, N, d, gmask, base, ::Val{D}) where {D}
    @inbounds for r in 1i32:D
        out[r] = 0.0f0
    end
    @inbounds for k in 1i32:D
        ck = N[d, k]                       # shared read, lane-local row d
        src = base + k
        for r in 1i32:D
            out[r] += shfl_sync(gmask, M[r], src) * ck
        end
    end
    return nothing
end

# out = (M · X)[:, d]   with M register col-per-lane, x = X[:, d] register col.
# (M·X)[r,d] = Σ_k M[r,k]·x[k]. D² shuffles.
@inline function _regmul_regcol!(out, M, x, gmask, base, ::Val{D}) where {D}
    @inbounds for r in 1i32:D
        out[r] = 0.0f0
    end
    @inbounds for k in 1i32:D
        ck = x[k]
        src = base + k
        for r in 1i32:D
            out[r] += shfl_sync(gmask, M[r], src) * ck
        end
    end
    return nothing
end

# In-place Cholesky on register column r (S = U'U, r becomes column d of U).
# Ported directly from the in-place dual-access Cholesky in src/operations.jl:
# A[i,j] -> r[i], pivot/diagonal shared across lanes via shfl_sync.
@inline function _reg_cholesky!(r, d, base, ::Val{D}) where {D}
    @inbounds for i in 1i32:D
        width = D - (i - 1i32)
        mask = ((UInt32(1) << (width % UInt32)) - UInt32(1)) << (((i - 1i32) + base) % UInt32)
        if d >= i
            Ai = r[i]
            for k in 1i32:(i - 1i32)
                Ak = r[k]
                Aki = shfl_sync(mask, Ak, base + i)   # U[k,i] from lane i
                Ai -= Aki * Ak
            end
            if d == i
                Ai = sqrt(Ai)
            end
            Aii = shfl_sync(mask, Ai, base + i)        # U[i,i] from lane i
            if d > i
                Ai = Ai / Aii
            end
            r[i] = Ai
        end
    end
    return nothing
end

# In-place forward solve  L y = rhs  with L = U' (lower), U in register col rU.
# L[i,j] = U[j,i] = element j of column i = rU[j] held by lane (base+i).
@inline function _forward_solve!(rhs, rU, gmask, base, ::Val{D}) where {D}
    @inbounds for i in 1i32:D
        yi = rhs[i]
        for j in 1i32:(i - 1i32)
            Lij = shfl_sync(gmask, rU[j], base + i)    # = U[j,i]
            yi -= Lij * rhs[j]                          # rhs[j] solved (j < i)
        end
        Lii = shfl_sync(gmask, rU[i], base + i)
        rhs[i] = yi / Lii
    end
    return nothing
end

# In-place backward solve  U x = rhs  with U (upper) in register col rU.
# U[i,j] = element i of column j = rU[i] held by lane (base+j).
@inline function _backward_solve!(rhs, rU, gmask, base, ::Val{D}) where {D}
    @inbounds for i in D:-1i32:1i32
        xi = rhs[i]
        for j in (i + 1i32):D
            Uij = shfl_sync(gmask, rU[i], base + j)    # = U[i,j]
            xi -= Uij * rhs[j]                          # rhs[j] solved (j > i)
        end
        Uii = shfl_sync(gmask, rU[i], base + i)
        rhs[i] = xi / Uii
    end
    return nothing
end

# out = (I - B1'·H)[:, d]   with B1 in register col r1, H shared.
# (B1'·H)[r,d] = Σ_k B1[k,r]·H[k,d];  B1[k,r] gathered from lane (base+r).
@inline function _i_minus_B1tH!(out, r1, H, d, gmask, base, ::Val{D}) where {D}
    @inbounds for r in 1i32:D
        s = 0.0f0
        src = base + r
        for k in 1i32:D
            b = shfl_sync(gmask, r1[k], src)           # B1[k,r]
            s += b * H[k, d]
        end
        out[r] = (r == d ? 1.0f0 : 0.0f0) - s
    end
    return nothing
end

# Coalesced global -> register-column load.
#
# A warp owns a contiguous run of `warp_region = n_mats_per_warp*D*D` global
# elements. We read them coalesced (lane `lid0` reads warp-elements
# {p*32 + lid0}), then shuffle-transpose into the column layout where lane
# `lid0` ends up holding warp-elements {lid0*D .. lid0*D+D-1} = its column.
#
# All 32 lanes participate (the transpose spans the whole warp), so this runs
# *outside* the per-matrix compute guard with a full-warp mask.
@inline function _coalesced_load_to_col!(
    r, Ps, warp_global_base::Int32, warp_first_mat::Int32, warp_region::Int32,
    n_passes::Int32, lid::Int32, N::Int32, ::Val{D},
) where {D}
    D2 = D * D
    lid0 = lid - 1i32
    tmp = MVector{Int(D),Float32}(undef)   # n_passes <= D slots used

    @inbounds for p in 0i32:(n_passes - 1i32)
        q = p * 32i32 + lid0
        gm = warp_first_mat + q ÷ D2
        tmp[p + 1i32] = (q < warp_region && gm <= N) ? Ps[warp_global_base + q + 1i32] : 0.0f0
    end

    # Transpose: r[i] (row i of this lane's column) = warp-element q = lid0*D + i,
    # which lane (q%32) read in pass (q÷32). Offered register must be uniform
    # across the warp, so we loop the offered pass `p` and store conditionally.
    @inbounds for p in 0i32:(n_passes - 1i32)
        offered = tmp[p + 1i32]
        for i in 0i32:(D - 1i32)
            q = lid0 * D + i
            v = shfl_sync(0xffffffff, offered, (q % 32i32) + 1i32)
            if q ÷ 32i32 == p
                r[i + 1i32] = v
            end
        end
    end
    return nothing
end

# Register-column -> coalesced global store (inverse of the load).
# Lane `lid0` writes warp-element `qp = p*32 + lid0` in pass `p`; that element
# lives in register (qp%D) of lane (qp÷D), gathered with a uniform offered
# register loop.
@inline function _coalesced_store_from_col!(
    Ps, r, warp_global_base::Int32, warp_first_mat::Int32, warp_region::Int32,
    n_passes::Int32, lid::Int32, N::Int32, ::Val{D},
) where {D}
    D2 = D * D
    lid0 = lid - 1i32

    @inbounds for p in 0i32:(n_passes - 1i32)
        qp = p * 32i32 + lid0
        srclane0 = qp ÷ D
        acc = 0.0f0
        for i in 0i32:(D - 1i32)
            v = shfl_sync(0xffffffff, r[i + 1i32], srclane0 + 1i32)
            if qp % D == i
                acc = v
            end
        end
        gm = warp_first_mat + qp ÷ D2
        if qp < warp_region && gm <= N
            Ps[warp_global_base + qp + 1i32] = acc
        end
    end
    return nothing
end

function kernel_kalman_register!(
    Ps_out,
    Ps_in,
    A_global,
    Q_global,
    H_global,
    R_global,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    base = (warp_matrix_id - 1i32) * D
    gmid = (bid - 1i32) * n_mats_per_block + (wid - 1i32) * n_mats_per_warp + warp_matrix_id

    # Shuffle mask for the D lanes of this matrix within the warp.
    gmask = ((UInt32(1) << (D % UInt32)) - UInt32(1)) << (base % UInt32)

    # Coalesced-transfer geometry: contiguous block of global elements this warp owns.
    warp_first_mat = (bid - 1i32) * n_mats_per_block + (wid - 1i32) * n_mats_per_warp + 1i32
    warp_global_base = (warp_first_mat - 1i32) * D * D
    warp_region = n_mats_per_warp * D * D

    # Number of passes needed to load/store the matrices for one warp
    n_passes = cld(warp_region, 32i32)

    # ---- batch-shared matrices A, Q, H, R in shared memory ----
    pad_interval = div(32i32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_Q = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_H = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_R = CuStaticSharedArray(Float32, (shmem_size_fixed,))

    if wid == 1i32
        shared_matrix_load!(shmem_A, A_global, Val(D))
    elseif wid == 2i32
        shared_matrix_load!(shmem_Q, Q_global, Val(D))
    elseif wid == 3i32
        shared_matrix_load!(shmem_H, H_global, Val(D))
    elseif wid == 4i32
        shared_matrix_load!(shmem_R, R_global, Val(D))
    end
    sync_threads()

    A = SharedMatrix(shmem_A, Val(D))
    Q = SharedMatrix(shmem_Q, Val(D))
    H = SharedMatrix(shmem_H, Val(D))
    R = SharedMatrix(shmem_R, Val(D))

    # Three register slots holding column d of three working matrices.
    # Declared for ALL lanes: the coalesced transpose below spans the whole warp.
    r1 = MVector{Int(D),Float32}(undef)
    r2 = MVector{Int(D),Float32}(undef)
    r3 = MVector{Int(D),Float32}(undef)

    # Coalesced load of P, transposed into the column layout (P -> r3).
    _coalesced_load_to_col!(
        r3, Ps_in, warp_global_base, warp_first_mat, warp_region, n_passes, lid, N, Val(D),
    )

    if warp_matrix_id <= n_mats_per_warp && gmid <= N
        ######################
        #### PREDICT STEP ####
        ######################
        _shared_mul_col!(r2, A, r3, Val(D))                  # r2 = A * P
        _regmul_sharedT!(r1, r2, A, d, gmask, base, Val(D))  # r1 = (A*P) * A'
        @inbounds for i in 1i32:D                            # r3 = A*P*A' + Q = P_pred
            r3[i] = r1[i] + Q[i, d]
        end

        #####################
        #### KALMAN GAIN ####
        #####################
        _shared_mul_col!(r1, H, r3, Val(D))                  # r1 = H * P_pred
        _regmul_sharedT!(r2, r1, H, d, gmask, base, Val(D))  # r2 = (H*P_pred) * H'
        @inbounds for i in 1i32:D                            # r2 = S = H*P_pred*H' + R
            r2[i] += R[i, d]
        end
        _reg_cholesky!(r2, d, base, Val(D))                  # r2 = U, S = U'U
        _forward_solve!(r1, r2, gmask, base, Val(D))         # r1 = U' \ (H*P_pred)
        _backward_solve!(r1, r2, gmask, base, Val(D))        # r1 = U \ . = K' = S^{-1} H P_pred

        #####################
        #### UPDATE STEP ####
        #####################
        _i_minus_B1tH!(r2, r1, H, d, gmask, base, Val(D))    # r2 = I - K*H   (K = (K')')
        _regmul_regcol!(r1, r2, r3, gmask, base, Val(D))     # r1 = (I-K*H) * P_pred = P_new
    end

    # Reconverge the warp (compute branch diverged), then coalesced store of P_new.
    sync_warp()
    _coalesced_store_from_col!(
        Ps_out, r1, warp_global_base, warp_first_mat, warp_region, n_passes, lid, N, Val(D),
    )

    return nothing
end

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, _, ::Val{:ours_register})
    D, _, N = size(P_in_cpu)

    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    nthreads = 2^8
    n_mats_per_block = (nthreads ÷ 32) * (32 ÷ D)
    nblocks = cld(N, n_mats_per_block)

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_kalman_register!(
            $P_out,
            $P_in,
            $A,
            $Q,
            $H,
            $R,
            Val(Int32($D)),
            Val(Int32($nthreads)),
            Int32($N),
        )
    end

    return median(bench_results.times) / 1e9 / N
end

# -----------------------------------------------------------------------------
# Correctness check against a CPU reference (run manually):
#   julia> include("benchmarking/kalman/ours_register.jl"); verify_register_kalman()
# -----------------------------------------------------------------------------
function verify_register_kalman(; D::Int=8, N::Int=1000, seed::Int=1)
    rng = MersenneTwister(seed)
    P_in = zeros(Float32, D, D, N)
    for i in 1:N
        Pi = rand(rng, Float32, D, D) / Float32(D)
        P_in[:, :, i] = Pi * Pi' + 0.1f0 * I
    end
    A = rand(rng, Float32, D, D) / Float32(D)
    Qe = rand(rng, Float32, D, D) / Float32(D)^2; Q = Qe * Qe' + 0.01f0 * I
    H = rand(rng, Float32, D, D) / Float32(D)
    Re = rand(rng, Float32, D, D) / Float32(D)^2; R = Re * Re' + 0.01f0 * I

    P_out = zeros(Float32, D, D, N)
    dP_out = cu(P_out); dP_in = cu(P_in)
    dA = cu(A); dQ = cu(Matrix{Float32}(Q)); dH = cu(H); dR = cu(Matrix{Float32}(R))

    nthreads = 2^8
    n_mats_per_block = (nthreads ÷ 32) * (32 ÷ D)
    nblocks = cld(N, n_mats_per_block)
    @cuda threads = nthreads blocks = nblocks kernel_kalman_register!(
        dP_out, dP_in, dA, dQ, dH, dR, Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
    )
    CUDA.synchronize()
    got = Array(dP_out)

    maxerr = 0.0f0
    for i in 1:N
        P = P_in[:, :, i]
        P_pred = A * P * A' + Q
        S = H * P_pred * H' + R
        K = P_pred * H' * inv(S)
        ref = (I - K * H) * P_pred
        maxerr = max(maxerr, maximum(abs.(got[:, :, i] .- ref)) / maximum(abs.(ref)))
    end
    println("D=$D N=$N  max relative error = $maxerr")
    return maxerr
end
