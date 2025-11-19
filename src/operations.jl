export batch_op!

using StaticArrays
using LinearAlgebra

@inline function batch_op!(
    ::typeof(+),
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
    ::Val{:small},
) where {T,D}
    for i in (1i32):D
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

    C[i, j] = A[i, j] + B[i, j]

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
    for i in (1i32):D
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

    C[i, j] = A[i, j] - B[i, j]

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
    ::Val{D},
    ::Val{:large},
    ::Val{:conseq},
) where {T,D}
    tid = threadIdx().x

    mat_elem_idx = mod1(tid, D * D)

    d = (mat_elem_idx - 1i32) ÷ D + 1i32
    i = mod1(mat_elem_idx, D)

    tot = zero(T)
    @inbounds for k in 1i32:D
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
    @inbounds for k in 1i32:D
        tot += A[i, k] * B[k, d]
    end

    @inbounds C[i, d] = tot

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

        if j >= i
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

        if j >= i
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
    for i in D:(-1i32):(1i32)
        x[i] = A[i, d]

        # Subtract contributions from already-computed elements
        for j in (i + 1i32):D
            x[i] -= U[i, j] * x[j]
        end

        # Divide by diagonal element
        x[i] /= U[i, i]
    end

    # Write result back to C
    for i in (1i32):D
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
    for i in (1i32):D
        y[i] = A[i, d]

        # Subtract contributions from already-computed elements
        for j in (1i32):(i - 1i32)
            y[i] -= L[i, j] * y[j]
        end

        # Divide by diagonal element
        y[i] /= L[i, i]
    end

    # Write result back to C
    for i in (1i32):D
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

# Transpose: B = transpose(A)
@inline function batch_op!(
    ::typeof(transpose),
    B::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32, 
    ::Val{D},
    ::Val{:small},
) where {T,D}
    # Each thread reads column d of A and writes it as row d of B
    for i in (1i32):D
        B[d, i] = A[i, d]
    end

    return nothing
end
