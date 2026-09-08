# Out-of-place Float32 factorization variants. All D lanes in each active matrix
# group must call these bodies, including lanes without a logical output column.
# Inputs and outputs must have distinct storage owners. Entry/exit shared-memory
# synchronization belongs to the caller; recurrence state below is private register
# storage, so no internal shared-memory producer/consumer barrier is necessary.

"""
    variant_op!(Val(:cholesky_row), U, A, d, Val(N), Val(D))

Compute the upper Cholesky factor of an SPD Float32 matrix into a fresh raw output.
A and U support RowAccess; U's full logical lower triangle is explicitly zeroed.
Input triangular masking is honored through `ours`, but callers must provide the
upper triangle of an SPD matrix. This is not a forced-in-place variant.
"""
@inline function variant_op!(
    ::Val{:cholesky_row},
    U::AbstractMatrix{Float32},
    A::AbstractMatrix{Float32},
    d::Int32,
    ::Val{N},
    ::Val{D},
) where {N,D}
    _validate_compute_shape(Val(N), Val(N), Val(D))
    base = mod1(threadIdx().x, 32i32) - d
    # The caller's complete-group participation contract proves geometry.
    work = @inbounds RegisterMatrix{Float32}(Val(N), Val(N), Val(D), RowOriented(), base, d)
    @inbounds @unroll for i in (1i32):Int32(N)
        work.mv[i] = d <= Int32(N) ? ours(A, i, d, RowAccess()) : 0.0f0
    end
    @inbounds @unroll for i in (1i32):Int32(N)
        value = work.mv[i]
        @unroll for k in (1i32):Int32(N)
            if k < i
                pivot_entry = theirs(work, k, i)
                value -= pivot_entry * work.mv[k]
            end
        end
        if d == i
            value = sqrt(value)
        end
        # Every lane offers a defined value, and the diagonal's source participates.
        work.mv[i] = value
        diagonal = theirs(work, i, i)
        work.mv[i] = if d == i
            value
        elseif i < d <= Int32(N)
            value / diagonal
        else
            0.0f0
        end
    end
    if d <= Int32(N)
        @inbounds @unroll for i in (1i32):Int32(N)
            ours_write!(U, i, d, work.mv[i], RowAccess())
        end
    end
    return U
end

@inline _solve_forward(::Union{LowerTriangular,UnitLowerTriangular}) = true
@inline _solve_forward(::Union{UpperTriangular,UnitUpperTriangular}) = false
@inline _solve_forward(A::Union{Adjoint,Transpose}) = !_solve_forward(parent(A))
@inline _solve_unit(::Union{LowerTriangular,UpperTriangular}) = false
@inline _solve_unit(::Union{UnitLowerTriangular,UnitUpperTriangular}) = true
@inline _solve_unit(A::Union{Adjoint,Transpose}) = _solve_unit(parent(A))

"""
    variant_op!(Val(:solve_row), C, factor, B, d, Val(N), Val(P), Val(D))

Out-of-place triangular solve C = factor \\ B. B/C use RowAccess for logical N×P
matrices. The N×N factor uses group-uniform broadcasts and may have either physical
orientation. Supported factor wrappers are lower/upper, unit/nonunit triangular,
and recursive outer adjoints/transposes of those wrappers. Every D-wide group lane
executes factor broadcasts even when d > P. Inputs must not alias C.
"""
@inline function variant_op!(
    ::Val{:solve_row},
    C::AbstractMatrix{Float32},
    factor::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
    d::Int32,
    ::Val{N},
    ::Val{P},
    ::Val{D},
) where {N,P,D}
    _validate_compute_shape(Val(N), Val(P), Val(D))
    forward = _solve_forward(factor)
    unit = _solve_unit(factor)
    work = MVector{Int(N),Float32}(undef)
    @inbounds @unroll for i in (1i32):Int32(N)
        work[i] = d <= Int32(P) ? ours(B, i, d, RowAccess()) : 0.0f0
    end
    # Column updates keep one solved scalar live while updating unsolved RHS entries.
    @inbounds @unroll for step in (1i32):Int32(N)
        i = forward ? step : Int32(N) - step + 1i32
        value = work[i]
        if !unit
            value /= theirs(factor, i, i)
        end
        if d <= Int32(P)
            ours_write!(C, i, d, value, RowAccess())
        end
        @unroll for j in (1i32):Int32(N)
            if forward ? j > i : j < i
                coefficient = theirs(factor, j, i)
                work[j] -= coefficient * value
            end
        end
    end
    return C
end
