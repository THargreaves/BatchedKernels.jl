# =============================================================================
# Operator overloads — the tracer surface
# =============================================================================
#
# Each method:
#   1. Walks any stdlib wrappers around its args via `register_wrapped!`
#   2. Emits a CallNode for the primitive operation
#   3. Returns a fresh TraceMatrix carrying the new ref
#
# Return types are pure functions of input type parameters, so
# `Core.Compiler.return_type(f, ArgTypes)` resolves to a concrete type when the
# user-supplied scalar function is type-stable.

# --- Matrix * Matrix --------------------------------------------------------

function Base.:*(A::TraceMatrix{T,D1,D2}, B::TraceMatrix{T,D2,D3}) where {T,D1,D2,D3}
    out = emit_call!(A.tape, *, NodeRef[A.ref, B.ref], TraceMatrix{T,D1,D3})
    return TraceMatrix{T,D1,D3}(A.tape, out)
end

function Base.:*(
    A::TraceMatrix{T,D1,D2}, B::Adjoint{T,TraceMatrix{T,D3,D2}}
) where {T,D1,D2,D3}
    tape = A.tape
    Bref = register_wrapped!(tape, B)
    out = emit_call!(tape, *, NodeRef[A.ref, Bref], TraceMatrix{T,D1,D3})
    return TraceMatrix{T,D1,D3}(tape, out)
end

function Base.:*(
    A::Adjoint{T,TraceMatrix{T,D2,D1}}, B::TraceMatrix{T,D2,D3}
) where {T,D1,D2,D3}
    tape = B.tape
    Aref = register_wrapped!(tape, A)
    out = emit_call!(tape, *, NodeRef[Aref, B.ref], TraceMatrix{T,D1,D3})
    return TraceMatrix{T,D1,D3}(tape, out)
end

# --- Matrix + Matrix --------------------------------------------------------

function Base.:+(A::TraceMatrix{T,D,D}, B::TraceMatrix{T,D,D}) where {T,D}
    out = emit_call!(A.tape, +, NodeRef[A.ref, B.ref], TraceMatrix{T,D,D})
    return TraceMatrix{T,D,D}(A.tape, out)
end

# --- I - Matrix -------------------------------------------------------------

function Base.:-(scaling::UniformScaling, M::TraceMatrix{T,D,D}) where {T,D}
    tape = M.tape
    sref = emit_const!(tape, scaling)
    out = emit_call!(tape, -, NodeRef[sref, M.ref], TraceMatrix{T,D,D})
    return TraceMatrix{T,D,D}(tape, out)
end

# --- cholesky ---------------------------------------------------------------

function LinearAlgebra.cholesky(A::TraceMatrix{T,D,D}) where {T,D}
    out = emit_call!(A.tape, cholesky, NodeRef[A.ref], TraceMatrix{T,D,D})
    factor = TraceMatrix{T,D,D}(A.tape, out)
    return Cholesky(factor, 'U', 0)
end

function LinearAlgebra.cholesky!(A::TraceMatrix{T,D,D}) where {T,D}
    out = emit_call!(A.tape, cholesky!, NodeRef[A.ref], TraceMatrix{T,D,D})
    factor = TraceMatrix{T,D,D}(A.tape, out)
    return Cholesky(factor, 'U', 0)
end

# --- Triangular solves ------------------------------------------------------

function Base.:\(
    L::LowerTriangular{T,S}, M::TraceMatrix{T,D,D}
) where {T,D,S<:AbstractMatrix{T}}
    tape = M.tape
    Lref = register_wrapped!(tape, L)
    out = emit_call!(tape, \, NodeRef[Lref, M.ref], TraceMatrix{T,D,D})
    return TraceMatrix{T,D,D}(tape, out)
end

function Base.:\(
    U::UpperTriangular{T,S}, M::TraceMatrix{T,D,D}
) where {T,D,S<:AbstractMatrix{T}}
    tape = M.tape
    Uref = register_wrapped!(tape, U)
    out = emit_call!(tape, \, NodeRef[Uref, M.ref], TraceMatrix{T,D,D})
    return TraceMatrix{T,D,D}(tape, out)
end

function LinearAlgebra.ldiv!(
    L::LowerTriangular{T,S}, M::TraceMatrix{T,D,D}
) where {T,D,S<:AbstractMatrix{T}}
    tape = M.tape
    Lref = register_wrapped!(tape, L)
    out = emit_call!(tape, ldiv!, NodeRef[Lref, M.ref], TraceMatrix{T,D,D})
    return TraceMatrix{T,D,D}(tape, out)
end

function LinearAlgebra.ldiv!(
    U::UpperTriangular{T,S}, M::TraceMatrix{T,D,D}
) where {T,D,S<:AbstractMatrix{T}}
    tape = M.tape
    Uref = register_wrapped!(tape, U)
    out = emit_call!(tape, ldiv!, NodeRef[Uref, M.ref], TraceMatrix{T,D,D})
    return TraceMatrix{T,D,D}(tape, out)
end
