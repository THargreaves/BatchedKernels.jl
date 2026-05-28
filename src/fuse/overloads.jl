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
#
# Dimension naming follows the matmul convention: for an (M×N) by (N×P)
# multiplication, the per-operand dims are D_M, D_N, D_P. Square ops collapse
# to a single dim (named D_M by convention).

# --- Matrix * Matrix --------------------------------------------------------

function Base.:*(
    A::TraceMatrix{T,D_M,D_N}, B::TraceMatrix{T,D_N,D_P}
) where {T,D_M,D_N,D_P}
    out = emit_call!(A.tape, *, NodeRef[A.ref, B.ref], TraceMatrix{T,D_M,D_P})
    return TraceMatrix{T,D_M,D_P}(A.tape, out)
end

function Base.:*(
    A::TraceMatrix{T,D_M,D_N}, B::Adjoint{T,TraceMatrix{T,D_P,D_N}}
) where {T,D_M,D_N,D_P}
    tape = A.tape
    Bref = register_wrapped!(tape, B)
    out = emit_call!(tape, *, NodeRef[A.ref, Bref], TraceMatrix{T,D_M,D_P})
    return TraceMatrix{T,D_M,D_P}(tape, out)
end

function Base.:*(
    A::Adjoint{T,TraceMatrix{T,D_N,D_M}}, B::TraceMatrix{T,D_N,D_P}
) where {T,D_M,D_N,D_P}
    tape = B.tape
    Aref = register_wrapped!(tape, A)
    out = emit_call!(tape, *, NodeRef[Aref, B.ref], TraceMatrix{T,D_M,D_P})
    return TraceMatrix{T,D_M,D_P}(tape, out)
end

# --- Matrix + Matrix --------------------------------------------------------

function Base.:+(
    A::TraceMatrix{T,D_M,D_N}, B::TraceMatrix{T,D_M,D_N}
) where {T,D_M,D_N}
    out = emit_call!(A.tape, +, NodeRef[A.ref, B.ref], TraceMatrix{T,D_M,D_N})
    return TraceMatrix{T,D_M,D_N}(A.tape, out)
end

# --- I - Matrix -------------------------------------------------------------

function Base.:-(scaling::UniformScaling, M::TraceMatrix{T,D_M,D_M}) where {T,D_M}
    tape = M.tape
    sref = emit_const!(tape, scaling)
    out = emit_call!(tape, -, NodeRef[sref, M.ref], TraceMatrix{T,D_M,D_M})
    return TraceMatrix{T,D_M,D_M}(tape, out)
end

# --- cholesky ---------------------------------------------------------------

function LinearAlgebra.cholesky(A::TraceMatrix{T,D_M,D_M}) where {T,D_M}
    out = emit_call!(A.tape, cholesky, NodeRef[A.ref], TraceMatrix{T,D_M,D_M})
    factor = TraceMatrix{T,D_M,D_M}(A.tape, out)
    return Cholesky(factor, 'U', 0)
end

function LinearAlgebra.cholesky!(A::TraceMatrix{T,D_M,D_M}) where {T,D_M}
    out = emit_call!(A.tape, cholesky!, NodeRef[A.ref], TraceMatrix{T,D_M,D_M})
    factor = TraceMatrix{T,D_M,D_M}(A.tape, out)
    return Cholesky(factor, 'U', 0)
end

# --- Triangular solves ------------------------------------------------------

function Base.:\(
    L::LowerTriangular{T,S}, M::TraceMatrix{T,D_M,D_N}
) where {T,D_M,D_N,S<:AbstractMatrix{T}}
    tape = M.tape
    Lref = register_wrapped!(tape, L)
    out = emit_call!(tape, \, NodeRef[Lref, M.ref], TraceMatrix{T,D_M,D_N})
    return TraceMatrix{T,D_M,D_N}(tape, out)
end

function Base.:\(
    U::UpperTriangular{T,S}, M::TraceMatrix{T,D_M,D_N}
) where {T,D_M,D_N,S<:AbstractMatrix{T}}
    tape = M.tape
    Uref = register_wrapped!(tape, U)
    out = emit_call!(tape, \, NodeRef[Uref, M.ref], TraceMatrix{T,D_M,D_N})
    return TraceMatrix{T,D_M,D_N}(tape, out)
end

function LinearAlgebra.ldiv!(
    L::LowerTriangular{T,S}, M::TraceMatrix{T,D_M,D_N}
) where {T,D_M,D_N,S<:AbstractMatrix{T}}
    tape = M.tape
    Lref = register_wrapped!(tape, L)
    out = emit_call!(tape, ldiv!, NodeRef[Lref, M.ref], TraceMatrix{T,D_M,D_N})
    return TraceMatrix{T,D_M,D_N}(tape, out)
end

function LinearAlgebra.ldiv!(
    U::UpperTriangular{T,S}, M::TraceMatrix{T,D_M,D_N}
) where {T,D_M,D_N,S<:AbstractMatrix{T}}
    tape = M.tape
    Uref = register_wrapped!(tape, U)
    out = emit_call!(tape, ldiv!, NodeRef[Uref, M.ref], TraceMatrix{T,D_M,D_N})
    return TraceMatrix{T,D_M,D_N}(tape, out)
end

# --- Matrix * Vector --------------------------------------------------------

function Base.:*(
    A::TraceMatrix{T,D_M,D_N}, x::TraceVector{T,D_N}
) where {T,D_M,D_N}
    out = emit_call!(A.tape, *, NodeRef[A.ref, x.ref], TraceVector{T,D_M})
    return TraceVector{T,D_M}(A.tape, out)
end

function Base.:*(
    A::Adjoint{T,TraceMatrix{T,D_N,D_M}}, x::TraceVector{T,D_N}
) where {T,D_M,D_N}
    tape = x.tape
    Aref = register_wrapped!(tape, A)
    out = emit_call!(tape, *, NodeRef[Aref, x.ref], TraceVector{T,D_M})
    return TraceVector{T,D_M}(tape, out)
end

# --- Vector + / - Vector ----------------------------------------------------

function Base.:+(a::TraceVector{T,D_M}, b::TraceVector{T,D_M}) where {T,D_M}
    out = emit_call!(a.tape, +, NodeRef[a.ref, b.ref], TraceVector{T,D_M})
    return TraceVector{T,D_M}(a.tape, out)
end

function Base.:-(a::TraceVector{T,D_M}, b::TraceVector{T,D_M}) where {T,D_M}
    out = emit_call!(a.tape, -, NodeRef[a.ref, b.ref], TraceVector{T,D_M})
    return TraceVector{T,D_M}(a.tape, out)
end
