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
_is_scalar_op(types) = all(t -> t <: TraceScalar || t <: Number, types)

# Lower a scalar primitive to `$dest = fn(args...)`. The destination is the
# scalar's pre-initialised Julia local (allocated in codegen.jl), so the
# assignment writes through to the function-level binding.
_emit_scalar_assign(fn, dest::Symbol, args::Vector) = :($dest = $(Expr(:call, fn, args...)))

function emit_primitive(::typeof(*), dest::Symbol, args::Vector, types::Vector, D_MAX::Int)
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
            *, $dest, $A, $B, d, Val(Int32($D_M)), Val(Int32($D_N)), Val(Int32($D_MAX))
        ))
    else
        # Matmul: A is (D_M, D_N), B is (D_N, D_P) -> C is (D_M, D_P).
        D_M, D_N = shape(types[1])
        D_N2, D_P = shape(types[2])
        D_N == D_N2 || error("emit_primitive(*): contraction dim mismatch ($D_N vs $D_N2)")
        return :(batch_op!(
            *, $dest, $A, $B, d, Val(Int32($D_M)), Val(Int32($D_N)), Val(Int32($D_P))
        ))
    end
end

function emit_primitive(::typeof(+), dest::Symbol, args::Vector, types::Vector, D_MAX::Int)
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
        ))
    else
        D_M, D_N = shape(types[1])
        return :(batch_op!(
            +, $dest, $A, $B, d, Val(Int32($D_M)), Val(Int32($D_N)), Val(Int32($D_MAX))
        ))
    end
end

function emit_primitive(::typeof(/), dest::Symbol, args::Vector, types::Vector, D_MAX::Int)
    _is_scalar_op(types) && return _emit_scalar_assign(/, dest, args)
    m, n = shape(types[1])
    return :(variant_op!(
        Val(:divide_row),
        $dest,
        $(args[1]),
        $(args[2]),
        d,
        Val(Int32($m)),
        Val(Int32($n)),
        Val(Int32($D_MAX)),
    ))
end

function emit_primitive(
    ::typeof(one), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    m, n = shape(types[1])
    return :(variant_op!(
        Val(:identity_row),
        $dest,
        $(args[1]),
        d,
        Val(Int32($m)),
        Val(Int32($n)),
        Val(Int32($D_MAX)),
    ))
end

function emit_primitive(::typeof(-), dest::Symbol, args::Vector, types::Vector, D_MAX::Int)
    if length(args) == 1
        # Unary scalar negation.
        return :($dest = -($(args[1])))
    end
    if _is_scalar_op(types)
        return _emit_scalar_assign(-, dest, args)
    end
    # Matrix and vector subtraction share the existing arithmetic kernels.
    # I - M is represented separately as an IAddSubWrapped value.
    a, b = args
    dims = shape(types[1])
    D_M = dims[1]
    D_N = length(dims) == 1 ? 0 : dims[2]
    return :(batch_op!(
        -, $dest, $a, $b, d, Val(Int32($D_M)), Val(Int32($D_N)), Val(Int32($D_MAX))
    ))
end

function emit_primitive(
    ::typeof(cholesky), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    A, = args
    D_M = shape(types[1])[1]
    return :(batch_op!(
        cholesky, $dest, $A, d, Val(Int32($D_M)), Val(Int32($D_MAX)), warp_matrix_id
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
        cholesky, $A, d, Val(Int32($D_MAX)), Int32($(32 ÷ D_MAX)), warp_matrix_id
    ))
end

function emit_primitive(::typeof(\), dest::Symbol, args::Vector, types::Vector, D_MAX::Int)
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
        ))
    end
    D_M, D_N = shape(types[2])
    return :(batch_op!(
        \, $dest, $LU, $R, d, Val(Int32($D_M)), Val(Int32($D_N)), Val(Int32($D_MAX))
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
    return :(batch_op!(\, $LU, $M, d, Val(Int32($D_MAX))))
end

# `shape(T)` — extract row/col extents from a (possibly wrapped) trace type.
# Matrices return `(D_M, D_N)`, vectors return `(D_M,)`. Wrapped trace types
# delegate to their underlying type, with adjoint swapping for matrices.
shape(::Type{TraceMatrix{T,D_M,D_N}}) where {T,D_M,D_N} = (D_M, D_N)
shape(::Type{TraceVector{T,D_M}}) where {T,D_M} = (D_M,)
shape(::Type{<:TraceScalar}) = ()
shape(::Type{<:Adjoint{T,S}}) where {T,S} = reverse(shape(S))
shape(::Type{<:Transpose{T,S}}) where {T,S} = reverse(shape(S))
shape(::Type{<:UnitLowerTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:UnitUpperTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:LowerTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:UpperTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:Symmetric{T,S}}) where {T,S} = shape(S)
shape(::Type{<:IAddSubWrapped{T,D_M}}) where {T,D_M} = (D_M, D_M)

# =============================================================================
# QR
# =============================================================================
#
# `_alloc_vec` is a placeholder primitive whose only job is to give the
# planner a vector slot to allocate. The slot view (`V{idx}`) is constructed
# by the standard batched-vector prologue in codegen. No kernel code emitted.

function emit_primitive(
    ::typeof(_alloc_vec), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    return :(nothing)
end

# `qr` takes A and the alloc'd tau vector slot as args; writes R+reflectors
# into `dest` and tau values into `tau_view`.
#
# A4 scope: square only and `D == D_MAX` (no rectangular, no masked variant).
# The masked rectangular case would mirror the cholesky / matmul rectangular
# variants; not in scope.
function emit_primitive(::typeof(qr), dest::Symbol, args::Vector, types::Vector, D_MAX::Int)
    A, tau_view = args
    D_M, D_N = shape(types[1])
    D_M == D_N ||
        error("emit_primitive(qr): rectangular QR not yet supported (got $(D_M)×$(D_N))")
    D_M == D_MAX || error(
        "emit_primitive(qr): QR currently requires D == D_MAX (got D=$D_M, D_MAX=$D_MAX); mix-with-larger-matrix kernels are out of A4 scope",
    )
    return :(batch_op!(
        qr,
        $dest,
        $tau_view,
        $A,
        d,
        Val(Int32($D_M)),
        Val(Int32($D_M)),
        Val(Int32($D_MAX)),
        warp_matrix_id,
    ))
end

# `_qr_Q_multiply` — lazy Q-multiply. Args are (R, tau, B, Val(Adj)) where
# Adj is splice-inlined as a `Val{true}`/`Val{false}` literal via the
# `ConstNode` returned by `arg_kernel_expr`.
function emit_primitive(
    ::typeof(_qr_Q_multiply), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    R_view, tau_view, B_view, adj_val = args
    # types: (TraceMatrix R, TraceVector tau, TraceMatrix B, Val{Adj} const)
    D_R, _ = shape(types[1])
    B_D1, B_D2 = shape(types[3])
    D_R == D_MAX || error(
        "emit_primitive(_qr_Q_multiply): Q*B currently requires D == D_MAX (got D=$D_R, D_MAX=$D_MAX)",
    )
    return :(batch_op!(
        Val(:qr_Q_multiply),
        $adj_val,
        $dest,
        $R_view,
        $B_view,
        d,
        $tau_view,
        Val(Int32($D_R)),
        Val(Int32($D_R)),
        Val(Int32($B_D1)),
        Val(Int32($B_D2)),
        Val(Int32($D_MAX)),
        warp_matrix_id,
    ))
end

# =============================================================================
# Scalar-producing reductions
# =============================================================================
#
# Both backends use reductions over the complete D_MAX-wide matrix group.
# Logical padding contributes zero; the leader result is broadcast to every lane.
# This also handles full-warp groups without shifting UInt32 by 32 bits.

function emit_primitive(
    ::typeof(logdet), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    M, = args
    n = shape(types[1])[1]
    return :($dest = variant_logdet($M, d, Val(Int32($n)), Val(Int32($D_MAX))))
end

function emit_primitive(
    fn::Union{typeof(_triangular_logabs),typeof(_triangular_detsign)},
    dest::Symbol,
    args::Vector,
    types::Vector,
    D_MAX::Int,
)
    body =
        fn === _triangular_logabs ? :variant_triangular_logabs : :variant_triangular_detsign
    n = shape(types[1])[1]
    return :($dest = $body($(args[1]), d, Val(Int32($n)), Val(Int32($D_MAX))))
end

function emit_primitive(
    ::typeof(_norm_sq), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    v, = args
    D_M, = shape(types[1])
    return :($dest = variant_norm_sq($v, d, Val(Int32($D_M)), Val(Int32($D_MAX))))
end

function emit_primitive(
    ::typeof(dot), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    n, = shape(types[1])
    return :(
        $dest = variant_dot($(args[1]), $(args[2]), d, Val(Int32($n)), Val(Int32($D_MAX)))
    )
end

function emit_primitive(
    ::typeof(symmetric_part), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    n = shape(types[1])[1]
    return :(variant_op!(
        Val(:symmetric_row), $dest, $(args[1]), d, Val(Int32($n)), Val(Int32($D_MAX))
    ))
end

# The standalone stack also works with legacy dual storage. Multi-result block
# QR uses the hybrid planner because all result lifetimes start at its producer.
function emit_primitive(
    ::typeof(qr_upper_stack), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    candidates = orientation_variants(qr_upper_stack, types...)
    isempty(candidates) &&
        throw(ArgumentError("Unsupported QR stack shapes or element types"))
    # Legacy storage keeps the established row-owned body, `:qr_stack_col` (named for
    # its ColAccess contract: each lane owns a row). The column-owned body is hybrid-only.
    variant = only(v for v in candidates if v.id === :qr_stack_col)
    return emit_variant(variant, dest, args, types, D_MAX)
end
