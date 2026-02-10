export batch_op!, gram

using StaticArrays
using LinearAlgebra

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
    @inbounds for i in (1i32):D1
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
    @inbounds for i in (1i32):D1
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
    ::Val,
    ::Val{:small},
) where {T,D1,D2}
    if d > D1
        return nothing
    end
    # Extract column d of B into registers
    B_col = @MVector zeros(T, Int64(D2))
    @inbounds for k in (1i32):D2
        B_col[k] = B[k, d]
    end

    # Compute each element of column d of C
    @inbounds for i in (1i32):D1
        tot = zero(T)
        for k in (1i32):D2
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
    # @inbounds for k in (1i32):D1
    for k in (1i32):D1
        A_col[k] = A[k, d]
    end

    # Compute each element of column d of G 
    # @inbounds for i in (1i32):d #D2
    for i in (1i32):d #D2
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

    for i in (1i32):D1
        # Create mask for threads j >= i within this matrix
        base = UInt32((warp_matrix_id - 1i32) * D)
        width = UInt32(D1 - i + 1i32)
        mask = (UInt32(1) << width) - UInt32(1)
        mask = mask << (base + UInt32(i - 1i32))

        @inbounds if j >= i
            # Load element from column j, row i
            Uij = A[i, j]

            # RMOD STEP: subtract contributions from previous columns
            for k in (1i32):(i - 1i32)
                Ukj = U[k, j]  # Load from already-computed Cholesky factors
                # Share the diagonal/column value with other threads
                Uki = shfl_sync(mask, Ukj, i + (warp_matrix_id - 1i32) * D)
                Uij -= Uki * Ukj
            end

            # RDIV STEP
            if j == i
                Uij = sqrt(Uij)
            end
            # Share result with other threads
            Uii = shfl_sync(mask, Uij, i + (warp_matrix_id - 1i32) * D)
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

    if lid > active_lanes || d > D1
        return 0.0f0
    end

    i = d  # Each thread is responsible for the i-th row
    
    R_col = @MVector zeros(T, Int64(D2))
    @inbounds for j in 1i32:D2
        R_col[j] = A[i, j]
    end

    tau_storage = 0.0f0

    # Loop through all columns
    for j in 1i32:min(D1 - 1i32, D2)
        # Compute norm of j-th column

        # Create mask for threads i >= j
        base = (warp_matrix_id - 1i32) * D
        width = D1 - j + 1i32
        mask = (UInt32(1) << width) - UInt32(1)
        mask = mask << (base + j - 1i32)

        @inbounds if i >= j
            offset_start = (1i32 << (32i32 - CUDA.clz(width - 1i32))) >> 1i32
            
            # Calculate norm via reduction sum
            offset = offset_start
            norm = R_col[j] * R_col[j]
            while offset > 0
                add = shfl_down_sync(mask, norm, offset)
                if i + offset <= width + j - 1i32
                    norm += add
                end
                offset >>= 1
            end

            # Store each element in vector v as v_elem, in each thread
            sign = ifelse(R_col[j] >= zero(T), one(T), -one(T))
            v_elem = R_col[j] - ifelse(i == j, -sign * sqrt(norm), zero(T))
            
            # Normalise v so that v[1] = 1.0. This is necessary to fit v into lower triangular
            # part of R for storage and calculation of Q later, despite not necessary when calculating R
            v1 = shfl_sync(mask, v_elem, lid - i + j)
            v_elem /= v1

            # Calculate tau via reduction sum
            offset = offset_start
            tau = v_elem * v_elem
            while offset > 0
                add = shfl_down_sync(mask, tau, offset)
                if i + offset <= width + j - 1i32
                    tau += add
                end
                offset >>= 1
            end
            tau = 2i32 / tau

            # Broadcast the value of tau to all threads within the group
            tau = shfl_sync(mask, tau, lid - i + j)
            
            # Store tau to be returned
            if j == i
                tau_storage = tau
            end

            # Computing H = I - tau * v * v^T, A <- HA = A - tau * v * (v^T A)
            
            # Compute v^T A first
            # Thread that owns v_i computes v_i * A_{i,t}, etc.
            # Then use reduce sum to calculate each element in R
            # Then, compute A - v v^T A
            # Loop columns
            for t in j:D2
                w_t = v_elem * R_col[t]

                # Summing to get w_t via reduction sum
                offset = offset_start
                while offset > 0
                    add = shfl_down_sync(mask, w_t, offset)
                    if i + offset <= width + j - 1i32
                        w_t += add
                    end
                    offset >>= 1
                end
                # Broadcast the value of w_t to all threads within the group
                w_t = shfl_sync(mask, w_t, lid - i + j)
                R_col[t] -= tau * v_elem * w_t
            end

            # Store v in strictly lower triangular part of R
            if i > j
                R_col[j] = v_elem
            end
        end
    end

    @inbounds for j in 1i32:D2
        R[i, j] = R_col[j]
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
    @inbounds for j in 1i32:min(D1, D2)
        Q[i, j] = ifelse(i == j, one(T), zero(T))
    end
    
    # Loop through the H_k...
    for j in min(D1, D2):-1i32:1i32
        base = (warp_matrix_id - 1i32) * D
        width = D1 - j + 1i32
        mask = (UInt32(1) << width) - UInt32(1)
        mask = mask << (base + j - 1i32)

        @inbounds if i >= j
            tau_j = shfl_sync(mask, tau, lid - i + j)  # Get tau from 'leader' thread
            w_i = Q[j, i]  # first element in v = 1, not stored in R
            for t in (j + 1i32):D1
                w_i += R[t, j] * Q[t, i]
            end

            R_elem = ifelse(i == j, 1.0f0, R[i, j])
            for t in j:min(D1, D2)
                w_t = shfl_sync(mask, w_i, lid - i + t)
                Q[i, t] -= tau_j * R_elem * w_t
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
    @inbounds for j in 1i32:D1
        Q[i, j] = ifelse(i == j, one(T), zero(T))
    end
    
    # Loop through the H_k...
    for j in min(D1, D2):-1i32:1i32
        base = (warp_matrix_id - 1i32) * D
        width = D1 - j + 1i32
        mask = (UInt32(1) << width) - UInt32(1)
        mask = mask << (base + j - 1i32)

        @inbounds if i >= j
            tau_j = shfl_sync(mask, tau, lid - i + j)  # Get tau from 'leader' thread
            w_i = Q[j, i]  # first element in v = 1, not stored in R
            for t in (j + 1i32):D1
                w_i += R[t, j] * Q[t, i]
            end

            R_elem = ifelse(i == j, 1.0f0, R[i, j])
            for t in j:D1
                w_t = shfl_sync(mask, w_i, lid - i + t)
                Q[i, t] -= tau_j * R_elem * w_t
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
        for j in 1i32:B_D2
            C[i, j] = B[i, j]
        end
    end
    @inbounds if i > B_D1 && i <= D1
        for j in 1i32:B_D2
            C[i, j] = 0.0f0
        end
    end

    # Loop through the H_k...
    @inbounds for j in js_range(Val(adj), Val(min(D1 - 1i32, D2)))
        base = (warp_matrix_id - 1i32) * D
        width = D1 - j + 1i32
        mask = (UInt32(1) << width) - UInt32(1)
        mask = mask << (base + j - 1i32)
        
        tau_j = 0.0f0
        if i >= j
            tau_j = shfl_sync(mask, tau, lid - i + j)  # Get tau from 'leader' thread
        end

        w_i = C[j, i]  # first element in v = 1, not stored in R
        for t in (j + 1i32):D1
            w_i += R[t, j] * C[t, i]
        end

        R_elem = ifelse(i == j, 1.0f0, R[i, j])
        for t in 1i32:B_D2
            w_t = shfl_sync(mask, w_i, lid - i + t)
            if i >= j
                C[i, t] -= tau_j * R_elem * w_t
            end
        end
    end
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

    # Store column d in registers
    x = @MVector zeros(T, Int64(D1))

    # Backward substitution from bottom to top
    @inbounds for i in D1:(-1i32):(1i32)
        x[i] = A[i, d]

        # Subtract contributions from already-computed elements
        for j in (i + 1i32):D1
            x[i] -= U[i, j] * x[j]
        end

        # Divide by diagonal element
        x[i] /= U[i, i]
    end

    # Write result back to C
    @inbounds for i in (1i32):D1
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

    # Store column d in registers
    y = @MVector zeros(T, Int64(D1))

    # Forward substitution from top to bottom
    @inbounds for i in (1i32):D1
        y[i] = A[i, d]

        # Subtract contributions from already-computed elements
        for j in (1i32):(i - 1i32)
            y[i] -= L[i, j] * y[j]
        end

        # Divide by diagonal element
        y[i] /= L[i, i]
    end

    # Write result back to C
    @inbounds for i in (1i32):D1
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