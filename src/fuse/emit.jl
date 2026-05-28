# =============================================================================
# Codegen emission per primitive
# =============================================================================
#
# Each `emit_primitive` method returns the kernel `Expr` that performs the
# operation in shared memory, given a destination slot symbol and argument
# expressions (which may be wrapped by `arg_kernel_expr` in codegen).

function emit_primitive end

function emit_primitive(::typeof(*), dest::Symbol, args::Vector, types::Vector)
    A, B = args
    # Args may be wrapped (Adjoint) — use the *first* TraceMatrix in either to
    # get D. The trace_element_type of an Adjoint or wrapped trace type still
    # has a Float/Int dim available via `shape`.
    D = shape(types[1])[1]
    return :(batch_op!(*, $dest, $A, $B, d, Val(Int32($D)), Val(:small)))
end

function emit_primitive(::typeof(+), dest::Symbol, args::Vector, types::Vector)
    A, B = args
    D = shape(types[1])[1]
    return :(batch_op!(+, $dest, $A, $B, d, Val(Int32($D)), Val(:small)))
end

function emit_primitive(::typeof(-), dest::Symbol, args::Vector, types::Vector)
    # First arg is the UniformScaling literal (passed through as-is);
    # second is the matrix slot expression.
    _, M = args
    D = shape(types[2])[1]
    return :(_batch_op_I_minus!($dest, $M, d, Val(Int32($D))))
end

function emit_primitive(::typeof(cholesky), dest::Symbol, args::Vector, types::Vector)
    A, = args
    D = shape(types[1])[1]
    return :(batch_op!(
        cholesky,
        $dest,
        $A,
        d,
        Val(Int32($D)),
        Int32($(32 ÷ D)),
        warp_matrix_id,
        Val(:small),
    ))
end

function emit_primitive(::typeof(cholesky!), dest::Symbol, args::Vector, types::Vector)
    A, = args
    D = shape(types[1])[1]
    return :(batch_op!(
        cholesky, $A, d, Val(Int32($D)), Int32($(32 ÷ D)), warp_matrix_id, Val(:small)
    ))
end

function emit_primitive(::typeof(\), dest::Symbol, args::Vector, types::Vector)
    LU, M = args
    D = shape(types[2])[1]
    return :(batch_op!(\, $dest, $LU, $M, d, Val(Int32($D)), Val(:small)))
end

function emit_primitive(::typeof(ldiv!), dest::Symbol, args::Vector, types::Vector)
    LU, M = args
    D = shape(types[2])[1]
    return :(batch_op!(\, $LU, $M, d, Val(Int32($D)), Val(:small)))
end

# `shape(T)` — extract matrix shape from a (possibly wrapped) trace type.
shape(::Type{TraceMatrix{T,D1,D2}}) where {T,D1,D2} = (D1, D2)
shape(::Type{<:Adjoint{T,S}}) where {T,S} = reverse(shape(S))
shape(::Type{<:LowerTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:UpperTriangular{T,S}}) where {T,S} = shape(S)
shape(::Type{<:Symmetric{T,S}}) where {T,S} = shape(S)
