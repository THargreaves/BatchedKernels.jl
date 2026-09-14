# Out-of-place Float32/Float64 factorization variants. All D lanes in each active matrix
# group must call these bodies, including lanes without a logical output column.
# Inputs and outputs must have distinct storage owners. Entry/exit shared-memory
# synchronization belongs to the caller; recurrence state below is private register
# storage, so no internal shared-memory producer/consumer barrier is necessary.

"""
    variant_op!(Val(:cholesky_row), U, A, d, Val(N), Val(D))

Compute the upper Cholesky factor of an SPD Float32/Float64 matrix into a fresh raw output.
A and U support RowAccess; U's full logical lower triangle is explicitly zeroed.
Input triangular masking is honored through `ours`, but callers must provide the
upper triangle of an SPD matrix. This is not a forced-in-place variant.
"""
@inline function variant_op!(
    ::Val{:cholesky_row},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    d::Int32,
    ::Val{N},
    ::Val{D},
) where {T<:Union{Float32,Float64},N,D}
    _validate_compute_shape(Val(N), Val(N), Val(D))
    base = mod1(threadIdx().x, 32i32) - d
    # The caller's complete-group participation contract proves geometry.
    work = @inbounds RegisterMatrix{T}(Val(N), Val(N), Val(D), RowOriented(), base, d)
    @inbounds @unroll for i in (1i32):Int32(N)
        work.mv[i] = d <= Int32(N) ? ours(A, i, d, RowAccess()) : zero(T)
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
            zero(T)
        end
    end
    if d <= Int32(N)
        @inbounds @unroll for i in (1i32):Int32(N)
            ours_write!(U, i, d, work.mv[i], RowAccess())
        end
    end
    return U
end

# Keep the solved scalar observable to the GPU compiler in every participating
# lane, including lanes without an output column. Otherwise it can sink all math
# behind d <= P while materializing every full-group factor broadcast first, giving
# O(N^2) live coefficients. This empty asm consumes a register value; it is not a
# GPU synchronization instruction or a memory fence. A memory-only clobber does not
# constrain these register dependencies. Final resource checks remain necessary.
# CPU execution needs no GPU scheduling constraint.
@inline _keep_solved_value_live(::Float32) = nothing
CUDA.@device_override @inline function _keep_solved_value_live(value::Float32)
    Base.llvmcall(
        """
        call void asm sideeffect "", "f"(float %0)
        ret void
        """,
        Cvoid,
        Tuple{Float32},
        value,
    )
    return nothing
end

# Double-precision equivalent of the existing scalar-use constraint.
@inline _keep_solved_value_live(::Float64) = nothing
CUDA.@device_override @inline function _keep_solved_value_live(value::Float64)
    Base.llvmcall(
        """
        call void asm sideeffect "", "d"(double %0)
        ret void
        """,
        Cvoid,
        Tuple{Float64},
        value,
    )
    return nothing
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
executes factor broadcasts even when d > P. Inputs must not alias C. A scalar-use
compiler constraint retains each solved value's dependencies in those lanes; it
prevents the observed separation of all broadcasts from output-lane-only arithmetic.
This does not add a shared-memory fence or replace final compiled-resource checks.
"""
@inline function variant_op!(
    ::Val{:solve_row},
    C::AbstractMatrix{T},
    factor::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{N},
    ::Val{P},
    ::Val{D},
) where {T<:Union{Float32,Float64},N,P,D}
    _validate_compute_shape(Val(N), Val(P), Val(D))
    forward = _solve_forward(factor)
    unit = _solve_unit(factor)
    work = MVector{Int(N),T}(undef)
    @inbounds @unroll for i in (1i32):Int32(N)
        work[i] = d <= Int32(P) ? ours(B, i, d, RowAccess()) : zero(T)
    end
    # Column updates keep one solved scalar live while updating unsolved RHS entries.
    @inbounds @unroll for step in (1i32):Int32(N)
        i = forward ? step : Int32(N) - step + 1i32
        value = work[i]
        if !unit
            value /= theirs(factor, i, i)
        end
        _keep_solved_value_live(value)
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

"""
    variant_op!(Val(:solve_col), C, factor, B, d, Val(N), Val(P), Val(D))

Out-of-place triangular solve with lane d owning row d: factor, B and C all require
ColAccess. Each lane holds P private RHS entries. The pivot's owner solves its row,
then every D-wide group lane broadcasts the P solved values to update unsolved rows.
This uses N*P pivot broadcasts instead of the row body's triangular factor broadcasts.

Supports the same lower/upper, unit/nonunit and outer adjoint/transpose factor
wrappers as solve_row. P may exceed N provided both fit D. Lanes d > N still execute
all broadcasts with initialized dummy entries, but never access invalid owned rows.
Inputs and C must not alias. The recurrence uses private registers and shuffles;
caller entry/exit shared-memory fences remain required, with no internal shared
producer/consumer dependency. This is a distinct orientation contract, not a mirror
of the row body, and its suitability depends on transfer costs and final resources.
"""
@inline function variant_op!(
    ::Val{:solve_col},
    C::AbstractMatrix{T},
    factor::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::Int32,
    ::Val{N},
    ::Val{P},
    ::Val{D},
) where {T<:Union{Float32,Float64},N,P,D}
    _validate_compute_shape(Val(N), Val(P), Val(D))
    forward = _solve_forward(factor)
    unit = _solve_unit(factor)
    base = mod1(threadIdx().x, 32i32) - d
    # The caller's complete-group participation contract proves geometry.
    work = @inbounds RegisterMatrix{T}(Val(N), Val(P), Val(D), ColOriented(), base, d)
    @inbounds @unroll for p in (1i32):Int32(P)
        work.mv[p] = d <= Int32(N) ? ours(B, p, d, ColAccess()) : zero(T)
    end
    @inbounds @unroll for step in (1i32):Int32(N)
        i = forward ? step : Int32(N) - step + 1i32
        @unroll for p in (1i32):Int32(P)
            if d == i && !unit
                work.mv[p] /= ours(factor, i, d, ColAccess())
            end
            # All lanes participate; the pivot source has completed its local solve.
            pivot = theirs(work, i, p)
            if d <= Int32(N) && (forward ? d > i : d < i)
                work.mv[p] -= ours(factor, i, d, ColAccess()) * pivot
            end
        end
    end
    if d <= Int32(N)
        @inbounds @unroll for p in (1i32):Int32(P)
            ours_write!(C, p, d, work.mv[p], ColAccess())
        end
    end
    return C
end
