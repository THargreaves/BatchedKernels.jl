# =============================================================================
# Codegen emission per primitive
# =============================================================================
#
# Each `emit_primitive` method returns the kernel `Expr` that performs the
# operation in shared memory, given:
#   - `dest`: the destination slot's view symbol
#   - `args`: the kernel-side argument expressions for each operand (which may
#     be wrapped, e.g. `adjoint(M2)`, by `arg_kernel_expr` in codegen)
#   - `types`: the trace-time types of each operand (drives per-operand dim
#     extraction via `shape`)
#   - `D_MAX`: the slot/warp layout dim (the global maximum over all inputs);
#     operations whose per-operand dim is smaller use the masked sub-kernel
#     variant which iterates over their actual extent and no-ops the rest of
#     the lane.
#
# Dim naming follows the matmul convention: an (M×N) by (N×P) multiplication
# gives D_M, D_N, D_P. Square ops collapse to a single D_M.

function emit_primitive end

# Scalar-op dispatch: true when every operand is either a `TraceScalar` or a
# plain `Number` (so the result is a register-resident scalar, not a slot).
# `UniformScaling` is NOT `<: Number`, so the `I - M` case still routes to
# the matrix branch.
_is_scalar_op(types) = all(t -> t <: TraceScalar || t <: Number, types)

# Lower a scalar primitive to `$dest = fn(args...)`. The destination is the
# scalar's pre-initialised Julia local (allocated in codegen.jl), so the
# assignment writes through to the function-level binding.
_emit_scalar_assign(fn, dest::Symbol, args::Vector) =
    :($dest = $(Expr(:call, fn, args...)))

function emit_primitive(
    ::typeof(*), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    if _is_scalar_op(types)
        return _emit_scalar_assign(*, dest, args)
    end
    A, B = args
    if types[2] <: AbstractVector
        # Matvec: A is (D_M, D_N), x is (D_N,) -> y is (D_M,).
        D_M, D_N = shape(types[1])
        D_N == shape(types[2])[1] || error(
            "emit_primitive(*): matvec contraction dim mismatch ($D_N vs $(shape(types[2])[1]))",
        )
        return :(batch_op!(
            *,
            $dest,
            $A,
            $B,
            d,
            Val(Int32($D_M)),
            Val(Int32($D_N)),
            Val(Int32($D_MAX)),
            Val(:small),
        ))
    else
        # Matmul: A is (D_M, D_N), B is (D_N, D_P) -> C is (D_M, D_P).
        D_M, D_N = shape(types[1])
        D_N2, D_P = shape(types[2])
        D_N == D_N2 ||
            error("emit_primitive(*): contraction dim mismatch ($D_N vs $D_N2)")
        return :(batch_op!(
            *,
            $dest,
            $A,
            $B,
            d,
            Val(Int32($D_M)),
            Val(Int32($D_N)),
            Val(Int32($D_P)),
            Val(:small),
        ))
    end
end

function emit_primitive(
    ::typeof(+), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    if _is_scalar_op(types)
        return _emit_scalar_assign(+, dest, args)
    end
    A, B = args
    if types[1] <: AbstractVector
        D_M, = shape(types[1])
        return :(batch_op!(
            +,
            $dest,
            $A,
            $B,
            d,
            Val(Int32($D_M)),
            Val(Int32(0)),  # unused dispatch slot
            Val(Int32($D_MAX)),
            Val(:small),
        ))
    else
        D_M, D_N = shape(types[1])
        return :(batch_op!(
            +,
            $dest,
            $A,
            $B,
            d,
            Val(Int32($D_M)),
            Val(Int32($D_N)),
            Val(Int32($D_MAX)),
            Val(:small),
        ))
    end
end

function emit_primitive(
    ::typeof(-), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    if length(args) == 1
        # Unary scalar negation.
        return :($dest = -($(args[1])))
    end
    if _is_scalar_op(types)
        return _emit_scalar_assign(-, dest, args)
    end
    if types[1] <: AbstractVector && types[2] <: AbstractVector
        # Vector subtraction.
        a, b = args
        D_M, = shape(types[1])
        return :(batch_op!(
            -,
            $dest,
            $a,
            $b,
            d,
            Val(Int32($D_M)),
            Val(Int32(0)),
            Val(Int32($D_MAX)),
            Val(:small),
        ))
    end
    # I - M: first arg is the UniformScaling literal (passed through as-is);
    # second is the matrix slot expression. Always square.
    _, M = args
    D_M = shape(types[2])[1]
    return :(_batch_op_I_minus!($dest, $M, d, Val(Int32($D_M))))
end

function emit_primitive(
    ::typeof(cholesky), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    A, = args
    D_M = shape(types[1])[1]
    return :(batch_op!(
        cholesky,
        $dest,
        $A,
        d,
        Val(Int32($D_M)),
        Val(Int32($D_MAX)),
        warp_matrix_id,
        Val(:small),
    ))
end

function emit_primitive(
    ::typeof(cholesky!), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    A, = args
    D_M = shape(types[1])[1]
    # TODO(A8): the in-place cholesky sub-kernel has no masked variant in
    # operations.jl; until one exists this primitive only works when the
    # operand dim equals D_MAX.
    D_M == D_MAX || error(
        "emit_primitive(cholesky!): masked in-place cholesky not yet supported (D_M=$D_M, D_MAX=$D_MAX)",
    )
    return :(batch_op!(
        cholesky,
        $A,
        d,
        Val(Int32($D_MAX)),
        Int32($(32 ÷ D_MAX)),
        warp_matrix_id,
        Val(:small),
    ))
end

function emit_primitive(
    ::typeof(\), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    LU, R = args
    if types[2] <: AbstractVector
        # Triangular \ vector. Only LowerTriangular has a sub-kernel today;
        # UpperTriangular vector solves would need a new `batch_op!` variant
        # in operations.jl.
        types[1] <: LowerTriangular || error(
            "emit_primitive(\\): triangular-vector solve only implemented for LowerTriangular (got $(types[1]))",
        )
        D_M, = shape(types[2])
        # Middle `Val` in the sub-kernel signature is an unused placeholder.
        return :(batch_op!(
            \,
            $dest,
            $LU,
            $R,
            d,
            Val(Int32($D_M)),
            Val(Int32(0)),
            Val(Int32($D_MAX)),
            warp_matrix_id,
            Val(:small),
        ))
    end
    D_M, D_N = shape(types[2])
    return :(batch_op!(
        \,
        $dest,
        $LU,
        $R,
        d,
        Val(Int32($D_M)),
        Val(Int32($D_N)),
        Val(Int32($D_MAX)),
        Val(:small),
    ))
end

function emit_primitive(
    ::typeof(ldiv!), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    LU, M = args
    D_M, D_N = shape(types[2])
    # TODO(A8): the in-place triangular-solve sub-kernel has no masked variant
    # in operations.jl; until one exists this primitive only works when both
    # operand dims equal D_MAX.
    (D_M == D_MAX && D_N == D_MAX) || error(
        "emit_primitive(ldiv!): masked in-place ldiv! not yet supported (D_M=$D_M, D_N=$D_N, D_MAX=$D_MAX)",
    )
    return :(batch_op!(\, $LU, $M, d, Val(Int32($D_MAX)), Val(:small)))
end

# `shape(T)` — extract row/col extents from a (possibly wrapped) trace type.
# Matrices return `(D_M, D_N)`, vectors return `(D_M,)`. Wrapped trace types
# delegate to their underlying type, with adjoint swapping for matrices.
shape(::Type{TraceMatrix{T,D_M,D_N}}) where {T,D_M,D_N} = (D_M, D_N)
shape(::Type{TraceVector{T,D_M}}) where {T,D_M} = (D_M,)
shape(::Type{<:TraceScalar}) = ()
shape(::Type{<:Adjoint{T,S}}) where {T,S} = reverse(shape(S))
shape(::Type{<:LowerTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:UpperTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:Symmetric{T,S}}) where {T,S} = shape(S)

# =============================================================================
# Scalar-producing reductions
# =============================================================================
#
# Reductions go through a `batch_op!(Val(:…), …)` sub-kernel that uses
# `warp_reduce_sum` internally, leaving the canonical value in the leader lane
# of each warp-matrix group (other lanes hold partial garbage). We then
# `shfl_sync` the leader's value to all D lanes so the TraceScalar local is
# uniformly correct — downstream scalar arithmetic on `(dest)` then produces
# the right value on every lane.

# Emit the boilerplate: call `reduction_call`, build the per-warp-matrix mask
# from (D_op, D_MAX), shuffle the leader's value, assign to `dest`.
function _emit_warp_reduction_broadcast(
    reduction_call::Expr, dest::Symbol, D_op::Int, D_MAX::Int
)
    val_sym = gensym(:val)
    base_sym = gensym(:base)
    mask_sym = gensym(:mask)
    return quote
        $val_sym = $reduction_call
        $base_sym = (warp_matrix_id - 1i32) * $(Int32(D_MAX))
        $mask_sym =
            ((UInt32(1) << ($(Int32(D_op)) % UInt32)) - UInt32(1)) <<
            ($base_sym % UInt32)
        $dest = shfl_sync($mask_sym, $val_sym, ($base_sym + 1i32) % UInt32)
    end
end

function emit_primitive(
    ::typeof(logdet), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    M, = args
    D_M = shape(types[1])[1]
    reduction = :(batch_op!(
        Val(:log_det),
        UpperTriangular($M),
        d,
        Val(Int32($D_M)),
        Val(Int32($D_MAX)),
        warp_matrix_id,
        Val(:small),
    ))
    return _emit_warp_reduction_broadcast(reduction, dest, D_M, D_MAX)
end

function emit_primitive(
    ::typeof(_norm_sq), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    v, = args
    D_M, = shape(types[1])
    reduction = :(batch_op!(
        Val(:mahal_dist),
        $v,
        d,
        Val(Int32($D_M)),
        Val(Int32($D_MAX)),
        warp_matrix_id,
        Val(:small),
    ))
    return _emit_warp_reduction_broadcast(reduction, dest, D_M, D_MAX)
end
