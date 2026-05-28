export batch_op!, gram

using StaticArrays
using LinearAlgebra
using KernelAbstractions.Extras: @unroll

gram(A) = A' * A

@inline function batch_op!(
    ::typeof(+),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    @inbounds for i in (1i32):D
        C[i, d] = A[i, d] + B[i, d]
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(+),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:large},
) where {T,D}
    i = mod1(mat_elem_idx, D)
    j = (mat_elem_idx - 1i32) ÷ D + 1i32

    @inbounds C[i, j] = A[i, j] + B[i, j]

    return nothing
end

@inline function batch_op!(
    ::typeof(+),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val,
    ::Val{:small},
) where {T,D1,D2}
    if d > D2
        return nothing
    end
    @inbounds @unroll for i in (1i32):D1
        C[i, d] = A[i, d] + B[i, d]
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(-),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    @inbounds for i in (1i32):D
        C[i, d] = A[i, d] - B[i, d]
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(-),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val,
    ::Val{:small},
) where {T,D1,D2}
    if d > D2
        return nothing
    end
    @inbounds @unroll for i in (1i32):D1
        C[i, d] = A[i, d] - B[i, d]
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(-),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:large},
) where {T,D}
    i = mod1(mat_elem_idx, D)
    j = (mat_elem_idx - 1i32) ÷ D + 1i32

    @inbounds C[i, j] = A[i, j] - B[i, j]

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    # Extract column d of B into registers
    B_col = @MVector zeros(T, Int64(D))
    @inbounds for k in (1i32):D
        B_col[k] = B[k, d]
    end

    # Compute each element of column d of C
    @inbounds for i in (1i32):D
        tot = zero(T)
        for k in (1i32):D
            tot += A[i, k] * B_col[k]
        end
        C[i, d] = tot
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D3},
    ::Val{:small},
) where {T,D1,D2,D3}  # (D1,D2) x (D2,D3) -> (D1,D3) multiplication
    if d > D3
        return nothing
    end

    # Extract column d of B into registers
    B_col = ntuple(k -> @inbounds(B[Int32(k), d]), Val(Int(D2)))

    # Compute each element of column d of C
    @inbounds @unroll for i in (1i32):D1
        C[i, d] = sum(ntuple(k -> @inbounds(A[i, Int32(k)]) * B_col[k], Val(Int(D2))))
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    U::UpperTriangular{T,<:AbstractMatrix{T}},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{:small},
) where {T,D1,D2,D}  # (D1,D2) x trig(D2,D2) = (D1,D2)
    if d > D2
        return nothing
    end

    # Extract column d of U into registers
    # Only upper trig part needed, but for compiler efficiency, load whole
    U_col = ntuple(k -> @inbounds(U[Int32(k), d]), Val(Int(D2)))

    # Compute each element of column d of C
    @inbounds @unroll for i in (1i32):D1
        tot = zero(T)
        @unroll for k in 1i32:D2
            if k <= d
                tot += A[i, k] * U_col[k]
            end
        end
        C[i, d] = tot
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    L::LowerTriangular{T,<:AbstractMatrix{T}},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{:small},
) where {T,D1,D2,D}  # (D1,D2) x trig(D2,D2) = (D1,D2)
    if d > D2
        return nothing
    end

    # Extract column d of L into registers
    # Only lower trig part needed, but for compiler efficiency, load whole
    L_col = ntuple(k -> @inbounds(L[Int32(k), d]), Val(Int(D2)))

    # Compute each element of column d of C
    @inbounds @unroll for i in (1i32):D1
        tot = zero(T)
        # for k in d:D2
        @unroll for k in 1i32:D2
            if k >= d
                tot += A[i, k] * L_col[k]
            end
        end
        C[i, d] = tot
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    ::Val{D},
    ::Val{:large},
    ::Val{:conseq},
) where {T,D}
    tid = threadIdx().x

    mat_elem_idx = mod1(tid, D * D)

    d = (mat_elem_idx - 1i32) ÷ D + 1i32
    i = mod1(mat_elem_idx, D)

    tot = zero(T)
    @inbounds for k in (1i32):D
        tot += A[i, k] * B[k, d]
    end

    @inbounds C[i, d] = tot

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    ::Val{D},
    ::Val{:large},
    ::Val{:indep},
) where {T,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32

    n_cols_per_warp = max(1i32, prevpow(2i32, 32i32 ÷ D))
    lanes_per_slot = 32i32 ÷ n_cols_per_warp
    global_d = (wid - 1i32) * n_cols_per_warp + div(lid - 1i32, lanes_per_slot) + 1i32
    d = mod1(global_d, D)
    i = mod1(lid, lanes_per_slot)

    tot = zero(T)
    @inbounds for k in (1i32):D
        tot += A[i, k] * B[k, d]
    end

    @inbounds C[i, d] = tot

    return nothing
end

@inline function batch_op!(
    ::typeof(gram),
    G::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val,
    ::Val{:small},
) where {T,D1,D2}
    if d > D2
        return nothing
    end

    # A' -> (D2,D1)
    # A  -> (D1,D2)
    # G  -> (D2,D2)

    # Extract column d of A into registers
    A_col = @MVector zeros(T, Int64(D1))
    @inbounds for k in (1i32):D1
        A_col[k] = A[k, d]
    end

    # Compute each element of column d of G 
    @inbounds for i in (1i32):d
        tot = zero(T)
        for k in (1i32):D1
            tot += A[k, i] * A_col[k]
        end
        G[i, d] = tot
        G[d, i] = tot
    end

    return nothing
end

# Out-of-place Cholesky: U = cholesky(A)
@inline function batch_op!(
    ::typeof(cholesky),
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    n_mats_per_warp::Int32,
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes
        return nothing
    end

    # Compute mask for warp-level synchronization
    # The mask ensures threads within the same matrix stay synchronized
    j = d  # column this thread is responsible for

    for i in (1i32):D
        # Create mask for threads j >= i within this matrix
        mask = ((1 << (D - (i - 1))) - 1) << (i - 1)
        mask = mask << ((warp_matrix_id - 1) * D)

        @inbounds if j >= i
            # Load element from column j, row i
            Ai = A[i, j]

            # RMOD STEP: subtract contributions from previous columns
            for k in (1i32):(i - 1i32)
                Ak = U[k, j]  # Load from already-computed Cholesky factors
                # Share the diagonal/column value with other threads
                Aki = shfl_sync(mask, Ak, i + (warp_matrix_id - 1i32) * D)
                Ai -= Aki * Ak
            end

            # RDIV STEP
            if j == i
                Ai = sqrt(Ai)
            end
            # Share result with other threads
            Aii = shfl_sync(mask, Ai, i + (warp_matrix_id - 1i32) * D)
            if j > i
                Ai = Ai / Aii
            end

            # Write result
            U[i, j] = Ai
        end
    end

    return nothing
end

# Out-of-place Cholesky: U = cholesky(A)
@inline function batch_op!(
    ::typeof(cholesky),
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D1,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes || d > D1
        return nothing
    end

    # Compute mask for warp-level synchronization
    # The mask ensures threads within the same matrix stay synchronized
    j = d  # column this thread is responsible for

    @unroll for i in (1i32):D1
        # Create mask for threads j >= i within this matrix
        base = (warp_matrix_id - 1i32) * D
        width = D1 - i + 1i32
        mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
        mask = mask << ((base + i - 1i32) % UInt32)

        @inbounds if j >= i
            # Load element from column j, row i
            Uij = A[i, j]

            # RMOD STEP: subtract contributions from previous columns
            # @unroll for k in (1i32):(i - 1i32)
            @unroll for k in 1i32:D1
                if k < i
                    Ukj = U[k, j]  # Load from already-computed Cholesky factors
                    # Share the diagonal/column value with other threads
                    # Uki = shfl_idx_f32(mask, Ukj, i + (warp_matrix_id - 1i32) * D)
                    Uki = shfl_idx_f32(mask, Ukj, lid - j + i)
                    Uij -= Uki * Ukj
                end
            end

            # RDIV STEP
            if j == i
                Uij = sqrt(Uij)
            end
            # Share result with other threads
            # Uii = shfl_idx_f32(mask, Uij, i + (warp_matrix_id - 1i32) * D)
            Uii = shfl_idx_f32(mask, Uij, lid - j + i)
            if j > i
                Uij = Uij / Uii
            end

            # Write result
            U[i, j] = Uij
        end
    end

    return nothing
end

# In-place Cholesky: A = cholesky(A)
@inline function batch_op!(
    ::typeof(cholesky),
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    n_mats_per_warp::Int32,
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes
        return nothing
    end

    # Compute mask for warp-level synchronization
    j = d

    for i in (1i32):D
        mask = ((1 << (D - (i - 1))) - 1) << (i - 1)
        mask = mask << ((warp_matrix_id - 1) * D)

        @inbounds if j >= i
            Ai = A[i, j]

            # RMOD STEP
            for k in (1i32):(i - 1i32)
                Ak = A[k, j]
                Aki = shfl_sync(mask, Ak, i + (warp_matrix_id - 1i32) * D)
                Ai -= Aki * Ak
            end

            # RDIV STEP
            if j == i
                Ai = sqrt(Ai)
            end
            Aii = shfl_sync(mask, Ai, i + (warp_matrix_id - 1i32) * D)
            if j > i
                Ai = Ai / Aii
            end

            A[i, j] = Ai
        end
    end

    return nothing
end

@inline function warp_reduce_sum(mask::UInt32, val::T, i::Int32, ::Val{width}) where {T,width}
    acc = val
    nsteps = 32i32 - leading_zeros(width - 1i32)
    @unroll for s in 1i32:nsteps
        offset = 1i32 << (nsteps - s)
        add = shfl_down_sync(mask, acc, offset)
        acc += ifelse(i + offset <= width, add, zero(T))
        # acc += add
    end
    return acc
end

# TODO: allow for max(D1, D2) < D, requires extra
@inline function batch_op!(
    ::typeof(qr),
    R::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D1,D2,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    # Padding (i > D1) rows do participate
    if lid > active_lanes
        return zero(T)
    end

    i = d
    is_padding_row = (i > D1)

    # Mask for all threads in a matrix, even padded ones
    base = (warp_matrix_id - 1i32) * D
    mask = ((UInt32(1) << (D % UInt32)) - UInt32(1)) << (base % UInt32)

    # Load matrix into registers, padding rows hold zeros throughout
    R_col = MVector{Int(D2),T}(undef)
    @inbounds @unroll for j in 1i32:D2
        R_col[j] = is_padding_row ? zero(T) : A[i, j]
    end

    tau_storage = zero(T)

    @unroll for j in 1i32:min(D1 - 1i32, D2)
        # 1. Compute norm of j-th column

        # This thread's contribution to the squared norm sum
        contrib_sq = ifelse(i >= j && !is_padding_row, R_col[j] * R_col[j], zero(T))

        # Compute norm via reduction sum
        norm_sq = warp_reduce_sum(mask, contrib_sq, i, Val(D))
        norm_sq = shfl_sync(mask, norm_sq, (base + 1i32) % UInt32)

        # 2. Householder vector and tau
        # Retrieve alpha = R[j, j] in every lane, from base + j
        alpha = shfl_sync(mask, R_col[j], (base + j) % UInt32)

        sign = ifelse(alpha >= zero(T), one(T), -one(T))
        beta = -sign * sqrt(norm_sq)
        v1 = alpha - beta

        # v_elem:   set to 0 for rows above j and padding rows
        #           alpha - beta for row j
        #           R_col[j] for rows below j.
        v_elem = ifelse(
            is_padding_row || i < j,
            zero(T),
            ifelse(
                i == j,
                v1,
                R_col[j]
            ),
        )
        # Normalise v so that v[1] = 1.0
        # This is necessary to fit v into lower triangular part of R
        # for storage and calculation of Q later, despite not necessary when
        # calculating R
        v_elem = v_elem / v1

        # tau's algebraic identity
        tau = (beta - alpha) / beta

        if j == i
            tau_storage = tau
        end

        tau_v_elem = tau * v_elem

        # 3. Computing H = I - tau * v * v^T, A <- HA = A - tau * v * (v^T A)

        # MAGMA trick:
        # 3.1: thread i writes its partial products into row i of scratch
        # scratch[i, t] = v[i] * R[i, t] * tau for each trailing column t.
        # Padding rows write zero because v_elem == 0 and R_col == 0.
        @inbounds @unroll for t in 1i32:D2
            if t >= j + 1i32
                R[i, t] = tau_v_elem * R_col[t]
            end
        end

        # Transpose: now thread i handles column i
        sync_warp(mask)

        # 3.2: thread i reads column i of scratch: R[r, i] for r in 1..D, and sums
        # This gives the i-th column's entry of tau * v^T A
        w_i = zero(T)
        @inbounds @unroll for r in 1i32:D
            if r >= j
                w_i += R[r, i]
            end
        end
        # w_i = i-th entry of tau * v^T A

        # Compute v * (tau * v^T A)
        @inbounds @unroll for t in 1i32:D2
            if t >= j + 1i32
                w_t = shfl_sync(mask, w_i, (base + t) % UInt32)
                R_col[t] -= v_elem * w_t
            end
        end

        # Store v in the strictly lower-triangular part of R_col[j] for rows i > j
        # For i == j we'll write beta below at the final write-back
        # For i < j leave R_col[j] alone since it holds previous-iter R values
        if (i > j) && !is_padding_row
            R_col[j] = v_elem
        elseif (i == j) && !is_padding_row
            R_col[j] = beta
        end
    end

    # Write back
    @inbounds @unroll for j in 1i32:D2
        if !is_padding_row
            R[i, j] = R_col[j]
        end
    end

    return tau_storage
end

@inline function batch_op!(
    ::Val{:qr_Q_thin},
    Q::AbstractMatrix{T},
    R::AbstractMatrix{T},
    d::Int32,
    tau::Float32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D1,D2,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D
    
    if lid > active_lanes || d > D1
        return nothing
    end
    
    i = d  # Each thread is responsible for the i-th row

    # Initialise Q as identity
    @inbounds @unroll for j in 1i32:min(D1, D2)
        Q[i, j] = ifelse(i == j, one(T), zero(T))
    end
    
    # Loop through the H_k...
    @unroll for j in min(D1, D2):-1i32:1i32
        base = (warp_matrix_id - 1i32) * D
        width = D1 - j + 1i32
        mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
        mask = mask << ((base + j - 1i32) % UInt32)

        @inbounds if i >= j
            tau_j = shfl_sync(mask, tau, max(lid - i + j, 1i32) % UInt32)  # Get tau from 'leader' thread
            w_i = Q[j, i]  # first element in v = 1, not stored in R
            # for t in (j + 1i32):D1
            @unroll for t in 1i32:D1
                if t > j
                    w_i += R[t, j] * Q[t, i]
                end
            end

            R_elem = ifelse(i == j, 1.0f0, R[i, j])
            tau_R_elem = tau_j * R_elem
            @unroll for t in 1i32:min(D1, D2)
                if t >= j
                    w_t = shfl_sync(mask, w_i, max(lid - i + t, 1i32) % UInt32)
                    Q[i, t] -= tau_R_elem * w_t
                end
            end
        end
    end
end

@inline function batch_op!(
    ::Val{:qr_Q_full},
    Q::AbstractMatrix{T},
    R::AbstractMatrix{T},
    d::Int32,
    tau::Float32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D1,D2,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D
    
    if lid > active_lanes || d > D1
        return nothing
    end
    
    i = d  # Each thread is responsible for the i-th row

    # Initialise Q as identity
    @inbounds @unroll for j in 1i32:D1
        Q[i, j] = ifelse(i == j, one(T), zero(T))
    end
    
    # Loop through the H_k...
    @unroll for j in min(D1, D2):-1i32:1i32
        base = (warp_matrix_id - 1i32) * D
        width = D1 - j + 1i32
        mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
        mask = mask << ((base + j - 1i32) % UInt32)

        @inbounds if i >= j
            tau_j = shfl_sync(mask, tau, max(lid - i + j, 1i32) % UInt32)  # Get tau from 'leader' thread
            w_i = Q[j, i]  # first element in v = 1, not stored in R
            # @unroll for t in (j + 1i32):D1
            @unroll for t in 1i32:D1
                if t > j
                    w_i += R[t, j] * Q[t, i]
                end
            end

            R_elem = ifelse(i == j, 1.0f0, R[i, j])
            tau_R_elem = tau_j * R_elem
            @unroll for t in 1i32:D1
                if t >= j
                    w_t = shfl_sync(mask, w_i, max(lid - i + t, 1i32) % UInt32)
                    Q[i, t] -= tau_R_elem * w_t
                end
            end
        end
    end
end

@inline js_range(::Val{true}, ::Val{K}) where {K} = (1i32:K)
@inline js_range(::Val{false}, ::Val{K}) where {K} = (K:-1i32:1i32)
"""
Computes C = QB or C = Q^T B using Householder transformations without materialising Q
"""
@inline function batch_op!(
    ::Val{:qr_Q_multiply},
    ::Val{adj},
    C::AbstractMatrix{T},
    R::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    tau::Float32,
    ::Val{D1},  # Rows of the original matrix that qr was called on, independent of whether Q is transposed or not
    ::Val{D2},  # Columns of the original matrix
    ::Val{B_D1},  # Rows of B
    ::Val{B_D2},  # Columns of B
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,adj,D1,D2,B_D1,B_D2,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D
    
    if lid > active_lanes || d > max(D1, B_D1, B_D2)
        return nothing
    end
    
    i = d

    @inbounds if C !== B && i <= B_D1
        @unroll for j in 1i32:B_D2
            C[i, j] = B[i, j]
        end
    end
    @inbounds if i > B_D1 && i <= D1
        @unroll for j in 1i32:B_D2
            C[i, j] = 0.0f0
        end
    end

    # Loop through the H_k...
    @inbounds @unroll for j in js_range(Val(adj), Val(min(D1 - 1i32, D2)))
        base = (warp_matrix_id - 1i32) * D
        width = D1 - j + 1i32
        mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
        mask = mask << ((base + j - 1i32) % UInt32)
        
        tau_j = zero(T)
        if i >= j
            tau_j = shfl_sync(mask, tau, max(lid - i + j, 1i32) % UInt32)  # Get tau from 'leader' thread
        end

        w_i = C[j, i]  # first element in v = 1, not stored in R
        @unroll for t in 1i32:D1
            if t > j
                w_i += R[t, j] * C[t, i]
            end
        end

        R_elem = ifelse(i == j, one(T), R[i, j])
        tau_R_elem = tau_j * R_elem
        @unroll for t in 1i32:B_D2
            w_t = shfl_sync(mask, w_i, max(lid - i + t, 1i32) % UInt32)
            C[i, t] -= ifelse(i >= j, tau_R_elem * w_t, zero(T))
        end
    end
end

@inline function warp_reduce_sum(mask::UInt32, val::T, i::Int32, width::Int32, ::Val{guard}) where {T,guard}
    acc = val
    nsteps = 32i32 - leading_zeros(width - 1i32)
    @unroll for s in 1i32:nsteps
        offset = 1i32 << (nsteps - s)
        add = shfl_down_sync(mask, acc, offset % UInt32)
        acc += ifelse(i + offset <= guard, add, zero(T))
    end
    return acc
end

@inline function shfl_idx_f32(mask::UInt32, val::Float32, src1::Int32)
    src0 = (src1 - 1i32) % UInt32
    ccall("llvm.nvvm.shfl.sync.idx.f32", llvmcall, Float32,
          (UInt32, Float32, UInt32, UInt32), mask, val, src0, 0x1f)
end

@inline function batch_op!(
    ::typeof(qr),
    R::AbstractMatrix{T},
    A::BlockMatrix_2_1{T},
    d::Int32,
    ::Val{D},
    ::Val{2},
    ::Val{1},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes
        return nothing
    end

    i = d

    # Mask for all threads in a matrix, even padded ones
    base = (warp_matrix_id - 1i32) * D
    mask = ((UInt32(1) << (D % UInt32)) - UInt32(1)) << (base % UInt32)

    # Load both halves into registers
    R_col_top = MVector{Int(D),T}(undef)
    R_col_bot = MVector{Int(D),T}(undef)
    @inbounds @unroll for j in 1i32:D
        R_col_top[j] = A[i, j]
        R_col_bot[j] = A[i + D, j]
    end

    @inbounds @unroll for j in 1i32:D
        # 1. Compute norm of j-th column

        # This thread's contribution to the squared norm sum
        contrib_sq = ifelse(
            i >= j,
            R_col_top[j] * R_col_top[j] + R_col_bot[j] * R_col_bot[j],
            R_col_bot[j] * R_col_bot[j]
        )

        # Compute norm via reduction sum
        norm_sq = warp_reduce_sum(mask, contrib_sq, i, Val(D))
        norm_sq = shfl_idx_f32(mask, norm_sq, lid - i + 1i32)

        # 2. Householder vector and tau
        # Retrieve alpha = R[j, j] in every lane, from base + j
        alpha = shfl_idx_f32(mask, R_col_top[j], lid - i + j)
        sign = ifelse(alpha >= zero(T), one(T), -one(T))
        beta = -sign * sqrt(norm_sq)
        v1 = alpha - beta

        # v_top:   set to 0 for rows above j and padding rows
        #           alpha - beta for row j
        #           R_col[j] for rows below j.
        v_top = ifelse(
            i < j,
            zero(T),
            ifelse(i == j, v1, R_col_top[j])
        )
        # Normalise v so that v[1] = 1.0
        # This is necessary to fit v into lower triangular part of R
        # for storage and calculation of Q later, despite not necessary when
        # calculating R
        v_top = v_top / v1

        # v_bot: always R_col_bot[j] / v1
        v_bot = R_col_bot[j] / v1

        # tau via algebraic identity
        tau = (beta - alpha) / beta

        tau_v_top = tau * v_top
        tau_v_bot = tau * v_bot

        # 3. Computing H = I - tau * v * v^T, A <- HA = A - tau * v * (v^T A)

        # MAGMA trick:
        # 3.1: thread i writes its partial products into row i of scratch
        # scratch[i, t] = v[i] * R[i, t] * tau for each trailing column t.
        # Padding rows write zero because v_elem == 0 and R_col == 0.
        @inbounds @unroll for t in 1i32:D
            if t >= j + 1i32
                R[i, t] = tau_v_top * R_col_top[t] + tau_v_bot * R_col_bot[t]
            end
        end

        # Transpose: now thread i handles column i
        sync_warp(mask)

        # 3.2: thread i reads column i of scratch: R[r, i] for r in 1..D, and sums
        # This gives the i-th column's entry of tau * v^T A
        w_i = zero(T)
        @inbounds @unroll for r in 1i32:D
            w_i += R[r, i]
        end
        # w_i = i-th entry of tau * v^T A

        # Compute v * (tau * v^T A)
        @inbounds @unroll for t in 1i32:D
            if t >= j + 1i32
                w_t = shfl_idx_f32(mask, w_i, base + t)
                R_col_top[t] -= v_top * w_t
                R_col_bot[t] -= v_bot * w_t
            end
        end

        # Store only the missing diagonal term
        # No need to store Householder vectors as Q is never needed to be materialised
        if i == j
            R_col_top[j] = beta
        end
    end

    # Write back
    @inbounds @unroll for j in 1i32:D
        if j >= i
            R[i, j] = R_col_top[j]
        end
    end

    return nothing
end

# Vector backed by a register MVector
struct RegVector{N,T} <: AbstractVector{T}
    v::MVector{N,T}
end
@inline Base.size(::RegVector{N}) where {N} = (N,)
@inline Base.@propagate_inbounds Base.getindex(rv::RegVector, j::Integer) = rv.v[j]
@inline Base.@propagate_inbounds Base.setindex!(rv::RegVector, val, j::Integer) = (rv.v[j] = val)

# Vector backed by row i of the R matrix
struct ShmemVector{M,T} <: AbstractVector{T}
    R::M
    i::Int32
end
@inline ShmemVector(R::AbstractMatrix{T}, i::Int32) where {T} = ShmemVector{typeof(R),T}(R, i)
@inline Base.size(sv::ShmemVector) = (size(sv.R, 2),)
@inline Base.@propagate_inbounds Base.getindex(sv::ShmemVector, j::Integer) = sv.R[sv.i, j]
@inline Base.@propagate_inbounds Base.setindex!(sv::ShmemVector, val, j::Integer) = (sv.R[sv.i, j] = val)

# In-place to bottom right block
@inline function batch_op!(
    ::typeof(qr),
    R::AbstractMatrix{T},
    A::BlockMatrixLowerTrig_2_2{T},
    d::Int32,
    ::Val{D},
    ::Val{THRESH},
    ::Val{2},  # Blocks vertically
    ::Val{2},  # Blocks horisontally
    warp_matrix_id::Int32,
    ::Val{:old},
) where {T,D,THRESH}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes || d > D
        return nothing
    end

    i = d

    if D >= THRESH
        R_BR = ShmemVector(R, i)
    else
        R_BR = RegVector(MVector{Int(D),T}(undef))
    end

    R_TL = MVector{Int(D),T}(undef)
    R_TR = MVector{Int(D),T}(undef)
    R_BL = MVector{Int(D),T}(undef)
    # R_BR = MVector{Int(D),T}(undef)
    @inbounds @unroll for j in 1i32:D
        R_TL[j] = A[i, j]
        R_TR[j] = zero(T)
        R_BL[j] = A[i + D, j]
        R_BR[j] = A[i + D, j + D]
    end


    # Phase 1: left half of 2D x 2D
    base = (warp_matrix_id - 1i32) * D
    mask = (UInt32(1) << (D % UInt32)) - UInt32(1)
    mask = mask << (base % UInt32)

    @inbounds @unroll for j in 1i32:D
        norm_sq = R_BL[j] * R_BL[j] + ifelse(i >= j, R_TL[j] * R_TL[j], zero(T))
        norm_sq = warp_reduce_sum(mask, norm_sq, i, Val(D))
        norm_sq = shfl_idx_f32(mask, norm_sq, lid - i + 1i32)

        sign = ifelse(R_TL[j] >= zero(T), one(T), -one(T))
        v_top = ifelse(i >= j, R_TL[j], zero(T)) - ifelse(i == j, -sign * sqrt(norm_sq), zero(T))
        v_bot = R_BL[j]

        v1 = shfl_idx_f32(mask, v_top, lid - i + j)
        v_top /= v1
        v_bot /= v1

        tau = v_bot * v_bot + ifelse(i >= j, v_top * v_top, zero(T))
        tau = warp_reduce_sum(mask, tau, i, Val(D))
        tau = T(2) / tau
        tau = shfl_idx_f32(mask, tau, lid - i + 1i32)

        # Apply to left half columns j, ..., D
        tau_v_top = tau * v_top
        tau_v_bot = tau * v_bot
        @unroll for t in 1i32:D
            if t >= j
                w_t = v_bot * R_BL[t] + ifelse(i >= j, v_top * R_TL[t], zero(T))
                w_t = warp_reduce_sum(mask, w_t, i, Val(D))
                w_t = shfl_idx_f32(mask, w_t, lid - i + 1i32)

                R_TL[t] -= ifelse(i >= j, tau_v_top * w_t, zero(T))
                R_BL[t] -= tau_v_bot * w_t
            end
        end

        # Apply to right half columns 1, ..., D
        @unroll for t in 1i32:D
            w_t = v_bot * R_BR[t] + ifelse(i >= j, v_top * R_TR[t], zero(T))
            w_t = warp_reduce_sum(mask, w_t, i, Val(D))
            w_t = shfl_idx_f32(mask, w_t, lid - i + 1i32)

            R_TR[t] -= ifelse(i >= j, tau_v_top * w_t, zero(T))
            R_BR[t] -= tau_v_bot * w_t
        end
    end

    # Phase 2: right half of 2D x 2D
    @inbounds @unroll for j in 1i32:(D - 1i32)
        width = D - j + 1i32
        mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
        mask = mask << ((base + j - 1i32) % UInt32)

        if i >= j
            norm_sq = R_BR[j] * R_BR[j]
            # norm_sq = warp_reduce_sum(mask, norm_sq, i, Val(D))
            norm_sq = warp_reduce_sum(mask, norm_sq, i, width, Val(D))

            sign = ifelse(R_BR[j] >= zero(T), one(T), -one(T))
            v_elem = R_BR[j] - ifelse(i == j, -sign * sqrt(norm_sq), zero(T))

            v1 = shfl_idx_f32(mask, v_elem, lid - i + j)
            v_elem /= v1

            tau = v_elem * v_elem
            tau = warp_reduce_sum(mask, tau, i, width, Val(D))
            tau = T(2) / tau
            tau = shfl_idx_f32(mask, tau, lid - i + j)
            tau_v_elem = tau * v_elem

            for t in 1i32:D
                if t >= j
                    w_t = v_elem * R_BR[t]
                    w_t = warp_reduce_sum(mask, w_t, i, width, Val(D))
                    w_t = shfl_idx_f32(mask, w_t, lid - i + j)
                    R_BR[t] -= tau_v_elem * w_t
                end
            end
        end
    end

    # Only write R_BR, as only that's needed for sqrt Kalman filter
    if D < THRESH
        @inbounds @unroll for j in 1i32:D
            R[i, j] = R_BR[j]
        end
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(qr),
    R::AbstractMatrix{T},
    A::BlockMatrixLowerTrig_2_2{T},
    d::Int32,
    ::Val{D},
    ::Val{THRESH},
    ::Val{2},
    ::Val{2},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D,THRESH}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes
        return nothing
    end

    i = d

    # Mask for all threads in a matrix, even padded ones
    base = (warp_matrix_id - 1i32) * D
    mask = ((UInt32(1) << (D % UInt32)) - UInt32(1)) << (base % UInt32)

    if D >= THRESH
        R_BR = ShmemVector(R, i)
    else
        R_BR = RegVector(MVector{Int(D),T}(undef))
    end

    R_TL = MVector{Int(D),T}(undef)
    R_TR = MVector{Int(D),T}(undef)
    R_BL = MVector{Int(D),T}(undef)

    @inbounds @unroll for j in 1i32:D
        R_TL[j] = A[i, j]
        R_TR[j] = zero(T)
        R_BL[j] = A[i + D, j]
        R_BR[j] = A[i + D, j + D]
    end

    @inbounds @unroll for j in 1i32:D
        # 1. Compute norm of j-th column

        # This thread's contribution to the squared norm sum
        contrib_sq = ifelse(
            i >= j,
            R_TL[j] * R_TL[j] + R_BL[j] * R_BL[j],
            R_BL[j] * R_BL[j],
        )

        # Compute norm via reduction sum
        norm_sq = warp_reduce_sum(mask, contrib_sq, i, Val(D))
        norm_sq = shfl_idx_f32(mask, norm_sq, lid - i + 1i32)

        # 2. Householder vector and tau
        # Retrieve alpha = R[j, j] in every lane, from base + j
        alpha = shfl_idx_f32(mask, R_TL[j], lid - i + j)
        sign = ifelse(alpha >= zero(T), one(T), -one(T))
        beta = -sign * sqrt(norm_sq)
        v1 = alpha - beta

        # v_top:   set to 0 for rows above j and padding rows
        #           alpha - beta for row j
        #           R_col[j] for rows below j.
        v_top = ifelse(
            i < j,
            zero(T),
            ifelse(i == j, v1, R_TL[j])
        )
        # Normalise v so that v[1] = 1.0
        # This is necessary to fit v into lower triangular part of R
        # for storage and calculation of Q later, despite not necessary when
        # calculating R
        v_top = v_top / v1

        # v_bot: always R_col_bot[j] / v1
        v_bot = R_BL[j] / v1

        # tau via algebraic identity
        tau = (beta - alpha) / beta
        tau_v_top = tau * v_top
        tau_v_bot = tau * v_bot

        # 3. Computing H = I - tau * v * v^T, A <- HA = A - tau * v * (v^T A)

        # MAGMA trick:
        # 3.1: thread i writes its partial products into row i of scratch
        # scratch[i, t] = v[i] * R[i, t] * tau for each trailing column t.
        # Padding rows write zero because v_elem == 0 and R_col == 0.
        @inbounds @unroll for t in 1i32:D
            if t >= j + 1i32
                # Write to bottom left, as in sqrt kalman, that can be corrupted
                A[i + D, t] = tau_v_top * R_TL[t] + tau_v_bot * R_BL[t]
            end
        end

        # Transpose: now thread i handles column i
        sync_warp(mask)

        # 3.2: thread i reads column i of scratch: R[r, i] for r in 1..D, and sums
        # This gives the i-th column's entry of tau * v^T A
        w_i_left = zero(T)
        @inbounds @unroll for r in 1i32:D
            w_i_left += A[r + D, i]
        end
        # w_i_left = i-th entry of tau * v^T A

        # Left stage: compute v * (tau * v^T A)
        @inbounds @unroll for t in 1i32:D
            if t >= j + 1i32
                w_t = shfl_idx_f32(mask, w_i_left, base + t)
                R_TL[t] -= v_top * w_t
                R_BL[t] -= v_bot * w_t
            end
        end

        # Set pivot element R_TL[j] = beta for diagonal lane
        if i == j
            R_TL[j] = beta
        end

        # Left stage: compute v * (tau * v^T A)
        @inbounds @unroll for t in 1i32:D
            A[i + D, t] = tau_v_top * R_TR[t] + tau_v_bot * R_BR[t]
        end
        sync_warp(mask)

        w_i_right = zero(T)
        @inbounds @unroll for r in 1i32:D
            w_i_right += A[r + D, i]
        end

        @inbounds @unroll for t in 1i32:D
            w_t = shfl_idx_f32(mask, w_i_right, base + t)
            R_TR[t] -= v_top * w_t
            R_BR[t] -= v_bot * w_t
        end
    end

    # Phase 2: right half of 2D x 2D
    @inbounds @unroll for j in 1i32:(D - 1i32)
        contrib_sq = ifelse(
            i >= j,
            R_BR[j] * R_BR[j],
            zero(T),
        )
        norm_sq = warp_reduce_sum(mask, contrib_sq, i, Val(D))
        norm_sq = shfl_idx_f32(mask, norm_sq, lid - i + 1i32)

        alpha = shfl_idx_f32(mask, R_BR[j], lid - i + j)
        sign = ifelse(alpha >= zero(T), one(T), -one(T))
        beta = -sign * sqrt(norm_sq)
        v1 = alpha - beta

        v_elem = ifelse(
            i < j,
            zero(T),
            ifelse(
                i == j,
                v1,
                R_BR[j],
            ),
        )

        v_elem = v_elem / v1

        tau = (beta - alpha) / beta
        tau_v_elem = tau * v_elem

        @inbounds @unroll for t in 1i32:D
            if t >= j + 1i32
                # Write to bottom left, as in sqrt kalman, it can be corrupted
                A[i + D, t] = tau_v_elem * R_BR[t]
            end
        end
        sync_warp(mask)

        w_i = zero(T)
        @inbounds @unroll for r in 1i32:D
            # Write to bottom left, as in sqrt kalman, it can be corrupted
            w_i += A[r + D, i]
        end

        @inbounds @unroll for t in 1i32:D
            if t >= j + 1i32
                w_t = shfl_idx_f32(mask, w_i, base + t)
                R_BR[t] -= v_elem * w_t
            end
        end

        if i == j
            R_BR[j] = beta
        end
    end

    if D < THRESH
        @inbounds @unroll for j in 1i32:D
            R[i, j] = R_BR[j]
        end
    end

    return nothing
end

# Out-of-place upper triangular backward solve: C = U \ A
@inline function batch_op!(
    ::typeof(\),
    C::AbstractMatrix{T},
    U::UpperTriangular{T,<:AbstractMatrix{T}},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    # Store column d in registers
    x = @MVector zeros(T, Int64(D))

    # Backward substitution from bottom to top
    @inbounds for i in D:(-1i32):(1i32)
        x[i] = A[i, d]

        # Subtract contributions from already-computed elements
        for j in (i + 1i32):D
            x[i] -= U[i, j] * x[j]
        end

        # Divide by diagonal element
        x[i] /= U[i, i]
    end

    # Write result back to C
    @inbounds for i in (1i32):D
        C[i, d] = x[i]
    end

    return nothing
end

# Out-of-place upper triangular backward solve: C = U \ A
@inline function batch_op!(
    ::typeof(\),
    C::AbstractMatrix{T},
    U::UpperTriangular{T,<:AbstractMatrix{T}},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val,
    ::Val{:small},
) where {T,D1,D2}
    if d > D2
        return nothing
    end

    x = MVector{Int(D1),T}(undef)

    @inbounds @unroll for i in D1:(-1i32):1i32
        xi = A[i, d]
        @unroll for j in 1i32:D1
            if j > i
                xi -= U[i, j] * x[j]
            end
        end
        x[i] = xi / U[i, i]
    end

    @inbounds @unroll for i in 1i32:D1
        C[i, d] = x[i]
    end

    return nothing
end

# In-place upper triangular backward solve wrapper: A = U \ A
@inline function batch_op!(
    ::typeof(\),
    U::UpperTriangular{T,<:AbstractMatrix{T}},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    return batch_op!(\, A, U, A, d, Val(D), Val(:small))
end

# Out-of-place lower triangular forward solve: C = L \ A
@inline function batch_op!(
    ::typeof(\),
    C::AbstractMatrix{T},
    L::LowerTriangular{T,<:AbstractMatrix{T}},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    # Store column d in registers
    y = @MVector zeros(T, Int64(D))

    # Forward substitution from top to bottom
    @inbounds for i in (1i32):D
        y[i] = A[i, d]

        # Subtract contributions from already-computed elements
        for j in (1i32):(i - 1i32)
            y[i] -= L[i, j] * y[j]
        end

        # Divide by diagonal element
        y[i] /= L[i, i]
    end

    # Write result back to C
    @inbounds for i in (1i32):D
        C[i, d] = y[i]
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(\),
    C::AbstractMatrix{T},
    L::LowerTriangular{T,<:AbstractMatrix{T}},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val,
    ::Val{:small},
) where {T,D1,D2}
    if d > D2
        return nothing
    end

    y = MVector{Int(D1),T}(undef)

    @inbounds @unroll for i in 1i32:D1
        yi = A[i, d]

        # @unroll for j in 1i32:(i - 1i32)
        @unroll for j in 1i32:D1
            if j < i
                yi -= L[i, j] * y[j]
            end
        end
        y[i] = yi / L[i, i]
    end

    @inbounds @unroll for i in 1i32:D1
        C[i, d] = y[i]
    end

    return nothing
end

# In-place lower triangular forward solve wrapper: A = L \ A
@inline function batch_op!(
    ::typeof(\),
    L::LowerTriangular{T,<:AbstractMatrix{T}},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    return batch_op!(\, A, L, A, d, Val(D), Val(:small))
end

# L c = b, solve for c
@inline function batch_op!(
    ::typeof(\),
    c::AbstractVector{T},
    L::LowerTriangular{T,<:AbstractMatrix{T}},
    b::AbstractVector{T},
    d::Int32,
    ::Val{D1},
    ::Val,
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D1,D}
    if d > D1
        return nothing
    end

    @inbounds c_d = b[d]

    base = (warp_matrix_id - 1i32) * D
    width = D1
    mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
    mask = mask << (base % UInt32)

    @inbounds @unroll for j in 1i32:D1
        c_d_div_L = c_d / L[j, j]
        c_j_div_L = shfl_sync(mask, c_d_div_L, (base + j) % UInt32)
        
        if d > j
            c_d -= L[d, j] * c_j_div_L
        end

        if d == j
            c_d = c_d_div_L
        end
    end

    c[d] = c_d

    return nothing
end

@inline function batch_op!(
    ::typeof(transpose),
    B::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32, 
    ::Val{D},
    ::Val{:small},
) where {T,D}
    if A === B
        # Each thread reads column d of A and writes it as row d of B
        @inbounds for i in (d + 1i32):D
            tmp = A[d, i]
            A[d, i] = A[i, d]
            A[i, d] = tmp
        end
    else
        @inbounds for i in (1i32):D
            B[d, i] = A[i, d]
        end
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(transpose),
    B::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32, 
    ::Val{D1},
    ::Val{D2},
    ::Val,
    ::Val{:small},
) where {T,D1,D2}
    if A === B
        if d > max(D1, D2)
            return nothing
        end

        @inbounds for i in (d + 1i32):min(D1, D2)
            tmp = A[d, i]
            A[d, i] = A[i, d]
            A[i, d] = tmp
        end

        if D1 < D2
            @inbounds for i in (D1 + 1i32):D2
                A[i, d] = A[d, i]
            end
        elseif D1 > D2
            @inbounds for i in (D2 + 1i32):D1
                A[d, i] = A[i, d]
            end
        end
    else
        if d > D2
            return nothing
        end

        @inbounds for i in (1i32):D1
            B[d, i] = A[i, d]
        end
    end

    return nothing
end

###########################
#### VECTOR OPERATIONS ####
###########################

@inline function batch_op!(
    ::typeof(*),
    y::AbstractVector{T},
    A::AbstractMatrix{T},
    x::AbstractVector{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes
        return nothing
    end

    tot = zero(T)
    @inbounds for k in (1i32):D
        tot += A[d, k] * x[k]
    end
    
    @inbounds y[d] = tot

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    y::AbstractVector{T},
    A::AbstractMatrix{T},
    x::AbstractVector{T},
    d::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{:small},
) where {T,D1,D2,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes || d > D1
        return nothing
    end

    tot = zero(T)
    @inbounds for k in (1i32):D2
        tot += A[d, k] * x[k]
    end
    
    @inbounds y[d] = tot

    return nothing
end

@inline function batch_op!(
    ::typeof(+),
    z::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes
        return nothing
    end

    @inbounds z[d] = x[d] + y[d]

    return nothing
end

@inline function batch_op!(
    ::typeof(+),
    z::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    d::Int32,
    ::Val{D1},
    ::Val,
    ::Val{D},
    ::Val{:small},
) where {T,D1,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes || d > D1
        return nothing
    end

    @inbounds z[d] = x[d] + y[d]

    return nothing
end

@inline function batch_op!(
    ::typeof(-),
    z::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes
        return nothing
    end

    @inbounds z[d] = x[d] - y[d]

    return nothing
end

@inline function batch_op!(
    ::typeof(-),
    z::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    d::Int32,
    ::Val{D1},
    ::Val,
    ::Val{D},
    ::Val{:small},
) where {T,D1,D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    if lid > active_lanes || d > D1
        return nothing
    end

    @inbounds z[d] = x[d] - y[d]

    return nothing
end

@inline function batch_op!(
    ::Val{:log_det},
    M::LinearAlgebra.AbstractTriangular{T},
    d::Int32,
    ::Val{D1},
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D1,D}
    if d > D1
        return zero(T)
    end

    base = (warp_matrix_id - 1i32) * D
    width = D1
    mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
    mask = mask << (base % UInt32)

    log_det = T(2) * warp_reduce_sum(mask, log(M[d, d]), d, Val(D1))
    
    return log_det
end


@inline function batch_op!(
    ::Val{:mahal_dist},
    v::AbstractVector{T},
    d::Int32,
    ::Val{D1},
    ::Val{D},
    warp_matrix_id::Int32,
    ::Val{:small},
) where {T,D1,D}
    if d > D1
        return zero(T)
    end

    base = (warp_matrix_id - 1i32) * D
    width = D1
    mask = (UInt32(1) << (width % UInt32)) - UInt32(1)
    mask = mask << (base % UInt32)

    mahal_dist = warp_reduce_sum(mask, v[d] * v[d], d, Val(D1))

    return mahal_dist
end

# Materialise `I - M` into a fresh slot. A5 in the merge plan will replace this
# with getter/setter wrappers that fold `I ± λM` into the consuming operation
# (no extra slot, no extra kernel).
@inline function _batch_op_I_minus!(
    C::AbstractMatrix{T}, M::AbstractMatrix{T}, d::Int32, ::Val{D}
) where {T,D}
    @inbounds for i in (Int32(1)):D
        C[i, d] = (i == d ? one(T) : zero(T)) - M[i, d]
    end
    return nothing
end