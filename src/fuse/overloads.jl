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

function Base.:*(A::TraceMatrix{T,D_M,D_N}, B::TraceMatrix{T,D_N,D_P}) where {T,D_M,D_N,D_P}
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

# Triangular state roots must retain their logical mask when multiplied. These
# methods record ordinary matmul nodes; the existing accessor/variant contracts
# handle the wrapper and its orientation without materializing a dense copy.
const TraceTriangularMatrix{T} = Union{
    UpperTriangular{T,<:TraceMatrix{T}},
    LowerTriangular{T,<:TraceMatrix{T}},
    UpperTriangular{T,<:Adjoint{T,<:TraceMatrix{T}}},
    LowerTriangular{T,<:Adjoint{T,<:TraceMatrix{T}}},
    UpperTriangular{T,<:Transpose{T,<:TraceMatrix{T}}},
    LowerTriangular{T,<:Transpose{T,<:TraceMatrix{T}}},
}
_trace_tape(A::TraceMatrix) = A.tape
_trace_tape(A) = _trace_tape(parent(A))
function _trace_wrapped_product(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    m, k = size(A)
    kb, n = size(B)
    k == kb || throw(DimensionMismatch("Matrix product inner dimensions differ"))
    tape = _trace_tape(A)
    tape === _trace_tape(B) ||
        throw(ArgumentError("Product operands belong to different tapes"))
    refs = NodeRef[register_wrapped!(tape, A), register_wrapped!(tape, B)]
    ref = emit_call!(tape, *, refs, TraceMatrix{T,m,n})
    return TraceMatrix{T,m,n}(tape, ref)
end
function Base.:*(A::TraceTriangularMatrix{T}, B::TraceMatrix{T}) where {T}
    return _trace_wrapped_product(A, B)
end
function Base.:*(A::TraceMatrix{T}, B::TraceTriangularMatrix{T}) where {T}
    return _trace_wrapped_product(A, B)
end
function Base.:*(A::TraceTriangularMatrix{T}, B::Adjoint{T,<:TraceMatrix{T}}) where {T}
    return _trace_wrapped_product(A, B)
end
function Base.:*(A::TraceTriangularMatrix{T}, B::Transpose{T,<:TraceMatrix{T}}) where {T}
    return _trace_wrapped_product(A, B)
end

# --- Matrix + Matrix --------------------------------------------------------

function Base.:+(A::TraceMatrix{T,D_M,D_N}, B::TraceMatrix{T,D_M,D_N}) where {T,D_M,D_N}
    out = emit_call!(A.tape, +, NodeRef[A.ref, B.ref], TraceMatrix{T,D_M,D_N})
    return TraceMatrix{T,D_M,D_N}(A.tape, out)
end

# --- Matrix - Matrix --------------------------------------------------------

function Base.:-(A::TraceMatrix{T,M,N}, B::TraceMatrix{T,M,N}) where {T,M,N}
    A.tape === B.tape ||
        throw(ArgumentError("Difference operands belong to different tapes"))
    out = emit_call!(A.tape, -, NodeRef[A.ref, B.ref], TraceMatrix{T,M,N})
    return TraceMatrix{T,M,N}(A.tape, out)
end

# Preserve ordinary wrapped addition, including the unmodified expression
# `(A + A') / 2`. The planner sees the adjoint and can reject unsafe aliasing.
for Wrapper in (Adjoint, Transpose)
    @eval begin
        function Base.:+(
            A::TraceMatrix{T,M,N}, B::$Wrapper{T,TraceMatrix{T,N,M}}
        ) where {T,M,N}
            A.tape === parent(B).tape ||
                throw(ArgumentError("Sum operands belong to different tapes"))
            ref = register_wrapped!(A.tape, B)
            out = emit_call!(A.tape, +, NodeRef[A.ref, ref], TraceMatrix{T,M,N})
            return TraceMatrix{T,M,N}(A.tape, out)
        end
        function Base.:+(
            A::$Wrapper{T,TraceMatrix{T,N,M}}, B::TraceMatrix{T,M,N}
        ) where {T,M,N}
            return B + A
        end
    end
end

# A genuine identity operation: the emitted body does not inspect A's values.
# In particular, NaN or Inf inputs must not contaminate off-diagonal zeros.
function Base.one(A::TraceMatrix{T,M,N}) where {T,M,N}
    M == N || throw(DimensionMismatch("multiplicative identity requires a square matrix"))
    out = emit_call!(A.tape, one, NodeRef[A.ref], TraceMatrix{T,M,N})
    return TraceMatrix{T,M,N}(A.tape, out)
end

function Base.:/(A::TraceMatrix{T,M,N}, b::Number) where {T,M,N}
    ref = emit_const!(A.tape, T(b))
    out = emit_call!(A.tape, /, NodeRef[A.ref, ref], TraceMatrix{T,M,N})
    return TraceMatrix{T,M,N}(A.tape, out)
end
function Base.:/(A::TraceMatrix{T,M,N}, b::TraceScalar{T}) where {T,M,N}
    A.tape === b.tape || throw(ArgumentError("Division operands belong to different tapes"))
    out = emit_call!(A.tape, /, NodeRef[A.ref, b.ref], TraceMatrix{T,M,N})
    return TraceMatrix{T,M,N}(A.tape, out)
end

function symmetric_part(A::TraceMatrix{T,D,D}) where {T<:Real,D}
    out = emit_call!(A.tape, symmetric_part, NodeRef[A.ref], TraceMatrix{T,D,D})
    return TraceMatrix{T,D,D}(A.tape, out)
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

# A factorization solve is a scalar composition of the existing triangular
# primitives. No separate device body or storage/orientation contract is needed.
function Base.:\(C::Cholesky{T,<:TraceMatrix{T,D,D}}, B::TraceMatrix{T,M,N}) where {T,D,M,N}
    D == M || throw(DimensionMismatch("Cholesky factor and RHS row dimensions differ"))
    C.uplo == 'U' ||
        throw(ArgumentError("Trace-backed Cholesky solves require upper-stored factors"))
    return C.U \ (C.L \ B)
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

function Base.:\(
    A::Union{UnitLowerTriangular{T,S},UnitUpperTriangular{T,S}}, M::TraceMatrix{T,D_M,D_N}
) where {T,D_M,D_N,S<:AbstractMatrix{T}}
    ref = register_wrapped!(M.tape, A)
    out = emit_call!(M.tape, \, NodeRef[ref, M.ref], TraceMatrix{T,D_M,D_N})
    return TraceMatrix{T,D_M,D_N}(M.tape, out)
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

function Base.:*(A::TraceMatrix{T,D_M,D_N}, x::TraceVector{T,D_N}) where {T,D_M,D_N}
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

# --- QR ---------------------------------------------------------------------
#
# `qr(A)` writes both the R+reflectors slot AND a per-householder-step tau
# vector. Modelled as two single-output CallNodes: an `_alloc_vec` placeholder
# that gives the planner a vector slot to allocate, and the qr CallNode that
# takes A and the alloc'd tau slot as args. The qr emit_primitive emits one
# kernel call that writes both slots. Downstream `Q*B` consumers reference
# both the R ref and the tau ref so the planner keeps their slots live.
#
# `:alloc_vec` is the placeholder opcode — empty function name; emit_primitive
# returns a no-op. The slot view is constructed by the standard batched-vector
# prologue in codegen.
#
# A4 scope: square A only. Rectangular QR is a future refinement.

function _alloc_vec end

struct QRResult{T,D}
    R::TraceMatrix{T,D,D}
    tau::TraceVector{T,D}
end

# Lazy Q operator: holds refs to R+reflectors and the tau vector; does not
# materialise Q. `Q*B` / `Q'*B` dispatch through explicit overloads below
# that emit `:qr_Q_multiply` CallNodes with the right adjoint flag.
struct LazyQTrace{T,D,Adj}
    R_ref::NodeRef
    tau_ref::NodeRef
    tape::Tape
end

# Internal opcode for the Q-multiply CallNode.
function _qr_Q_multiply end

function LinearAlgebra.qr(A::TraceMatrix{T,D,D}) where {T,D}
    tape = A.tape
    # `_alloc_vec` has no args, so `emit_call!` would compute lifecycle =
    # LITERAL (empty parent set) and the planner would skip slot allocation.
    # The semantic lifecycle of an alloc'd batched tau slot is BATCHED — set
    # it explicitly here via push_node!.
    tau_ref = push_node!(
        tape, CallNode(_alloc_vec, NodeRef[]), NodeMeta(TraceVector{T,D}, BATCHED)
    )
    R_ref = emit_call!(tape, qr, NodeRef[A.ref, tau_ref], TraceMatrix{T,D,D})
    R = TraceMatrix{T,D,D}(tape, R_ref)
    tau = TraceVector{T,D}(tape, tau_ref)
    return QRResult{T,D}(R, tau)
end

# `.R` returns an `UpperTriangular` view over the R+reflectors slot. The
# triangular wrapper masks the reflector entries below the diagonal so
# downstream triangular-solve / matmul consumers only see R.
#
# `.Q` returns a `LazyQTrace` (no kernel materialisation); subsequent `Q*B`
# / `Q'*B` dispatch via the explicit overloads below.
function Base.getproperty(q::QRResult{T,D}, s::Symbol) where {T,D}
    s === :R && return UpperTriangular(getfield(q, :R))
    s === :Q && return LazyQTrace{T,D,false}(
        getfield(q, :R).ref, getfield(q, :tau).ref, getfield(q, :R).tape
    )
    return getfield(q, s)
end

# Q' just flips the Adj flag; nothing to emit until a consumer hits.
function Base.adjoint(q::LazyQTrace{T,D,Adj}) where {T,D,Adj}
    return LazyQTrace{T,D,!Adj}(q.R_ref, q.tau_ref, q.tape)
end

# Q * B and Q' * B share this single overload — the Adj flag is encoded in
# the LazyQTrace's type parameter and emitted as a `ConstNode(Val(Adj))`
# arg that `emit_primitive(_qr_Q_multiply)` reads.
function Base.:*(q::LazyQTrace{T,D,Adj}, B::TraceMatrix{T,D,K}) where {T,D,Adj,K}
    adj_ref = emit_const!(q.tape, Val(Adj))
    out = emit_call!(
        q.tape,
        _qr_Q_multiply,
        NodeRef[q.R_ref, q.tau_ref, B.ref, adj_ref],
        TraceMatrix{T,D,K},
    )
    return TraceMatrix{T,D,K}(q.tape, out)
end

# --- cholesky(::Symmetric) --------------------------------------------------
#
# Stdlib's `logdet(::Symmetric)` lowers to `logdet(cholesky(A))`, so this needs
# to dispatch to our trace overload rather than to stdlib's generic cholesky
# which would iterate scalar entries.

function LinearAlgebra.cholesky(A::Symmetric{T,<:TraceMatrix{T,D_M,D_M}}) where {T,D_M}
    # TODO: support `uplo='L'`. The current cholesky sub-kernel only reads the
    # upper triangle of its input and writes the U-factor into the upper
    # triangle of its output, so naively stripping `Symmetric(M, :L)` would
    # silently read the wrong triangle. A proper fix requires either a
    # lower-triangle cholesky sub-kernel in `operations.jl` (full stdlib
    # parity), or feeding `adjoint(A.data)` through the existing kernel
    # (numerically correct via `.U`/`.L` but the returned Cholesky's raw
    # `.factors` layout would diverge from stdlib's convention). Erroring
    # for now keeps both correctness and external representation honest.
    A.uplo == 'U' || error(
        "BatchedKernels: cholesky(Symmetric(..., :L)) is not yet supported — " *
        "wrap with `Symmetric(.., :U)` (the default) or transpose your input " *
        "before wrapping.",
    )
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
    v.tape === w.tape || throw(ArgumentError("Dot operands belong to different tapes"))
    out = if v.ref == w.ref
        emit_call!(v.tape, _norm_sq, NodeRef[v.ref], TraceScalar{T})
    else
        emit_call!(v.tape, dot, NodeRef[v.ref, w.ref], TraceScalar{T})
    end
    return TraceScalar{T}(v.tape, out)
end

function LinearAlgebra.logdet(C::Cholesky{T,<:TraceMatrix{T,D_M,D_M}}) where {T,D_M}
    tape = C.factors.tape
    out = emit_call!(tape, logdet, NodeRef[C.factors.ref], TraceScalar{T})
    return TraceScalar{T}(tape, out)
end

function LinearAlgebra.logdet(M::Symmetric{T,<:TraceMatrix{T,D_M,D_M}}) where {T<:Real,D_M}
    return logdet(cholesky(M))
end

# Separate scalar nodes keep the ordinary tuple compatible with both planners.
# Neither operation assumes a positive diagonal, unlike covariance_root_logdet.
function _triangular_logabs end
function _triangular_detsign end

function LinearAlgebra.logabsdet(
    A::TraceTriangularMatrix{T}
) where {T<:Union{Float32,Float64}}
    tape = _trace_tape(A)
    ref = register_wrapped!(tape, A)
    magnitude = emit_call!(tape, _triangular_logabs, NodeRef[ref], TraceScalar{T})
    sign = emit_call!(tape, _triangular_detsign, NodeRef[ref], TraceScalar{T})
    return TraceScalar{T}(tape, magnitude), TraceScalar{T}(tape, sign)
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

# These identities depend only on the numerical type, not the runtime value.
Base.zero(::TraceScalar{T}) where {T} = zero(T)
Base.zero(::Type{TraceScalar{T}}) where {T} = zero(T)
Base.one(::TraceScalar{T}) where {T} = one(T)
Base.one(::Type{TraceScalar{T}}) where {T} = one(T)

# The straight-line tracer cannot capture runtime branches. In particular,
# Number's equality fallback can compare tracer identities and silently select
# a branch (including isnan's x != x test), so reject comparisons explicitly.
function _runtime_scalar_condition_error()
    return throw(
        ArgumentError(
            "Runtime scalar comparisons and conditions are unsupported inside fused functions; " *
            "use literal(value, n) for an intentional trace-time configuration constant",
        ),
    )
end
for op in (:(==), :isequal, :isless, :<, :<=)
    @eval begin
        Base.$op(::TraceScalar, ::TraceScalar) = _runtime_scalar_condition_error()
        Base.$op(::TraceScalar, ::Number) = _runtime_scalar_condition_error()
        Base.$op(::Number, ::TraceScalar) = _runtime_scalar_condition_error()
    end
end
for op in (:iszero, :isone, :isfinite, :isinf, :isnan)
    @eval Base.$op(::TraceScalar) = _runtime_scalar_condition_error()
end

for _op in (:+, :-, :*, :/)
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
        function Base.$_op(::TraceScalar, ::TraceScalar)
            throw(ArgumentError("Traced scalar arithmetic requires matching element types"))
        end
    end
end

function Base.:-(s::TraceScalar{T}) where {T}
    out = emit_call!(s.tape, -, NodeRef[s.ref], TraceScalar{T})
    return TraceScalar{T}(s.tape, out)
end
