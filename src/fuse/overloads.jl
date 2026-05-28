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

# --- aI + bM as a zero-cost wrapper -----------------------------------------
#
# `±sI ± M` (and `M ± sI`) emits an IAddSubWrapped NewNode eagerly and returns
# a `TraceMatrix{T,D_M,D_M}` whose `.ref` points at the wrapper. Consumers
# (matmul, etc.) dispatch through the ordinary `*(::TraceMatrix, ::TraceMatrix)`
# overload — no wrapper-specific overloads needed — and `arg_kernel_expr`
# lowers the NewNode to `IAddSubGetterMatrix(parent_view, a, b)`, folding the
# I-shift into the consuming sub-kernel.
#
# Scope: `b ∈ {+1, -1}` (driven by the `+` / `-` operator); `a = ±s.λ` from
# the `UniformScaling`. Generalising to `b = λ_M` (i.e. `I ± λM`) requires a
# `Number × TraceMatrix` overload — deferred with the rest of `Scalar × Matrix`.

function Base.:-(s::UniformScaling, M::TraceMatrix{T,D_M,D_M}) where {T,D_M}
    return _emit_iaddsub_wrap(M, T(s.λ), -one(T))
end

function Base.:-(M::TraceMatrix{T,D_M,D_M}, s::UniformScaling) where {T,D_M}
    return _emit_iaddsub_wrap(M, T(-s.λ), one(T))
end

function Base.:+(s::UniformScaling, M::TraceMatrix{T,D_M,D_M}) where {T,D_M}
    return _emit_iaddsub_wrap(M, T(s.λ), one(T))
end

function Base.:+(M::TraceMatrix{T,D_M,D_M}, s::UniformScaling) where {T,D_M}
    return _emit_iaddsub_wrap(M, T(s.λ), one(T))
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

# --- Triangular \ Vector ----------------------------------------------------

function Base.:\(
    L::LowerTriangular{T,S}, v::TraceVector{T,D_M}
) where {T,D_M,S<:AbstractMatrix{T}}
    tape = v.tape
    Lref = register_wrapped!(tape, L)
    out = emit_call!(tape, \, NodeRef[Lref, v.ref], TraceVector{T,D_M})
    return TraceVector{T,D_M}(tape, out)
end

# --- cholesky(::Symmetric) --------------------------------------------------
#
# Stdlib's `logdet(::Symmetric)` lowers to `logdet(cholesky(A))`, so this needs
# to dispatch to our trace overload rather than to stdlib's generic cholesky
# which would iterate scalar entries.

function LinearAlgebra.cholesky(A::Symmetric{T,<:TraceMatrix{T,D_M,D_M}}) where {T,D_M}
    return cholesky(A.data)
end

# --- Reductions to scalar ---------------------------------------------------
#
# Internal opcode for `‖v‖²`. Backed by the existing `:mahal_dist` sub-kernel
# (a one-vector squared-norm reduction); `dot(v, v)` and `sum(abs2, v)` share
# this single CallNode opcode.

function _norm_sq end

function Base.sum(::typeof(abs2), v::TraceVector{T,D_M}) where {T,D_M}
    out = emit_call!(v.tape, _norm_sq, NodeRef[v.ref], TraceScalar{T})
    return TraceScalar{T}(v.tape, out)
end

function LinearAlgebra.dot(v::TraceVector{T,D_M}, w::TraceVector{T,D_M}) where {T,D_M}
    v.ref == w.ref || error(
        "BatchedKernels: dot(u, v) with distinct tape values is not supported; use sum(abs2, v) / dot(v, v)",
    )
    out = emit_call!(v.tape, _norm_sq, NodeRef[v.ref], TraceScalar{T})
    return TraceScalar{T}(v.tape, out)
end

function LinearAlgebra.logdet(C::Cholesky{T,<:TraceMatrix{T,D_M,D_M}}) where {T,D_M}
    tape = C.factors.tape
    out = emit_call!(tape, logdet, NodeRef[C.factors.ref], TraceScalar{T})
    return TraceScalar{T}(tape, out)
end

function LinearAlgebra.logdet(
    M::Symmetric{T,<:TraceMatrix{T,D_M,D_M}}
) where {T<:Real,D_M}
    return logdet(cholesky(M))
end

# --- Scalar arithmetic ------------------------------------------------------
#
# Lane-replicated: every D lanes of the warp-matrix hold the canonical scalar
# value (a reduction's shfl-broadcast establishes that; subsequent arithmetic
# is per-lane). The 1/D utilisation is the cost of keeping the result a
# regular Julia local — no additional sync, no extra shmem.
#
# Number literals get folded as `ConstNode(T(x))` so they bake into the
# generated kernel at the trace eltype.

for _op in (:+, :-, :*)
    @eval begin
        function Base.$_op(a::TraceScalar{T}, b::TraceScalar{T}) where {T}
            out = emit_call!(a.tape, $_op, NodeRef[a.ref, b.ref], TraceScalar{T})
            return TraceScalar{T}(a.tape, out)
        end
        function Base.$_op(a::TraceScalar{T}, b::Number) where {T}
            bref = emit_const!(a.tape, T(b))
            out = emit_call!(a.tape, $_op, NodeRef[a.ref, bref], TraceScalar{T})
            return TraceScalar{T}(a.tape, out)
        end
        function Base.$_op(a::Number, b::TraceScalar{T}) where {T}
            aref = emit_const!(b.tape, T(a))
            out = emit_call!(b.tape, $_op, NodeRef[aref, b.ref], TraceScalar{T})
            return TraceScalar{T}(b.tape, out)
        end
    end
end

function Base.:-(s::TraceScalar{T}) where {T}
    out = emit_call!(s.tape, -, NodeRef[s.ref], TraceScalar{T})
    return TraceScalar{T}(s.tape, out)
end
