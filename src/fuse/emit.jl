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

function emit_primitive(
    ::typeof(*), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    A, B = args
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

function emit_primitive(
    ::typeof(+), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    A, B = args
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

function emit_primitive(
    ::typeof(-), dest::Symbol, args::Vector, types::Vector, D_MAX::Int
)
    # First arg is the UniformScaling literal (passed through as-is);
    # second is the matrix slot expression. I - M is always square.
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
    LU, M = args
    D_M, D_N = shape(types[2])
    return :(batch_op!(
        \,
        $dest,
        $LU,
        $M,
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

# `shape(T)` — extract matrix shape from a (possibly wrapped) trace type.
shape(::Type{TraceMatrix{T,D_M,D_N}}) where {T,D_M,D_N} = (D_M, D_N)
shape(::Type{<:Adjoint{T,S}}) where {T,S} = reverse(shape(S))
shape(::Type{<:LowerTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:UpperTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:Symmetric{T,S}}) where {T,S} = shape(S)
