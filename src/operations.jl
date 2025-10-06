export batch_op!

using StaticArrays
using LinearAlgebra

@inline function batch_op!(
    ::typeof(+),
    C::DualAccessMatrix{T},
    A::DualAccessMatrix{T},
    B::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
) where {T,D}
    for i in (1i32):D
        C[i, d] = A[i, d] + B[i, d]
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(-),
    C::DualAccessMatrix{T},
    A::DualAccessMatrix{T},
    B::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
) where {T,D}
    for i in (1i32):D
        C[i, d] = A[i, d] - B[i, d]
    end

    return nothing
end

@inline function batch_op!(
    ::typeof(*),
    C::DualAccessMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{D},
) where {T,D}
    # Extract column d of B into registers
    B_col = @MVector zeros(T, Int64(D))
    for k in (1i32):D
        B_col[k] = B[k, d]
    end

    # Compute each element of column d of C
    for i in (1i32):D
        tot = zero(T)
        for k in (1i32):D
            tot += A[i, k] * B_col[k]
        end
        C[i, d] = tot
    end

    return nothing
end

# Out-of-place Cholesky: U = cholesky(A)
@inline function batch_op!(
    ::typeof(cholesky),
    U::DualAccessMatrix{T},
    A::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
    n_mats_per_warp::Int32,
    warp_matrix_id::Int32,
) where {T,D}
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
    A::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
    n_mats_per_warp::Int32,
    warp_matrix_id::Int32,
) where {T,D}
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

# Upper triangular backward solve: C = U \ A
@inline function batch_op!(
    ::typeof(\),
    C::DualAccessMatrix{T},
    U::UpperTriangular{T,<:DualAccessMatrix{T}},
    A::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
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

# Lower triangular forward solve: C = L \ A
@inline function batch_op!(
    ::typeof(\),
    C::DualAccessMatrix{T},
    L::LowerTriangular{T,<:DualAccessMatrix{T}},
    A::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
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

# Transpose: B = transpose(A)
@inline function batch_op!(
    ::typeof(transpose),
    B::DualAccessMatrix{T},
    A::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
) where {T,D}
    # Each thread reads column d of A and writes it as row d of B
    for i in (1i32):D
        B[d, i] = A[i, d]
    end

    return nothing
end
