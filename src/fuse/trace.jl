# =============================================================================
# TraceMatrix and tracing infrastructure
# =============================================================================
#
# `TraceMatrix` is the trace-time representative of a batched scalar matrix.
# It subtypes `AbstractMatrix` so Julia's standard wrappers (`Adjoint`,
# `UpperTriangular`, ...) accept it — those build *real* stdlib wrapper objects
# around a phantom `TraceMatrix`. The wrapper structure is captured lazily by
# `register_wrapped!` at each consuming primitive.
#
# Scalar indexing is forbidden so user-side mistakes don't silently descend into
# stdlib's generic AbstractMatrix fallbacks.

struct TraceMatrix{T,D_M,D_N} <: AbstractMatrix{T}
    tape::Tape
    ref::NodeRef
end

Base.size(::TraceMatrix{T,D_M,D_N}) where {T,D_M,D_N} = (D_M, D_N)
Base.size(::TraceMatrix{T,D_M,D_N}, i::Int) where {T,D_M,D_N} =
    i == 1 ? D_M : (i == 2 ? D_N : 1)
Base.length(::TraceMatrix{T,D_M,D_N}) where {T,D_M,D_N} = D_M * D_N
Base.axes(::TraceMatrix{T,D_M,D_N}) where {T,D_M,D_N} = (Base.OneTo(D_M), Base.OneTo(D_N))
Base.IndexStyle(::Type{<:TraceMatrix}) = IndexCartesian()
Base.eltype(::Type{<:TraceMatrix{T}}) where {T} = T

Base.getindex(::TraceMatrix, ::Vararg) = error(
    "Scalar indexing on TraceMatrix is forbidden inside a vmapped function."
)
Base.setindex!(::TraceMatrix, _, ::Vararg) =
    error("setindex! on TraceMatrix is forbidden.")

# -----------------------------------------------------------------------------
# TraceVector
# -----------------------------------------------------------------------------
#
# Trace-time representative of a batched scalar vector. Like `TraceMatrix` but
# 1D — lives in a slot with the single-access layout only (no dual-access
# second buffer), so the load path skips the interm→dual transfer.

struct TraceVector{T,D_M} <: AbstractVector{T}
    tape::Tape
    ref::NodeRef
end

Base.size(::TraceVector{T,D_M}) where {T,D_M} = (D_M,)
Base.size(::TraceVector{T,D_M}, i::Int) where {T,D_M} = i == 1 ? D_M : 1
Base.length(::TraceVector{T,D_M}) where {T,D_M} = D_M
Base.axes(::TraceVector{T,D_M}) where {T,D_M} = (Base.OneTo(D_M),)
Base.IndexStyle(::Type{<:TraceVector}) = IndexLinear()
Base.eltype(::Type{<:TraceVector{T}}) where {T} = T

Base.getindex(::TraceVector, ::Vararg) = error(
    "Scalar indexing on TraceVector is forbidden inside a vmapped function."
)
Base.setindex!(::TraceVector, _, ::Vararg) =
    error("setindex! on TraceVector is forbidden.")

# -----------------------------------------------------------------------------
# TraceScalar
# -----------------------------------------------------------------------------
#
# Trace-time representative of a batched scalar. Lives in registers, replicated
# across the D lanes of a warp-matrix (broadcast via `shfl_sync` after each
# reduction). No shared-memory slot; planner tracks only liveness.
#
# Subtypes `Number` so `log(s)`, `s + 0.5`, etc. dispatch through normal Number
# overloads. Trace operations are defined explicitly so promotion does not
# materialise a TraceScalar where a real Number is expected.

struct TraceScalar{T} <: Number
    tape::Tape
    ref::NodeRef
end

Base.eltype(::Type{<:TraceScalar{T}}) where {T} = T

# -----------------------------------------------------------------------------
# Runtime-container → trace-element type map
# -----------------------------------------------------------------------------

function trace_element_type(::Type{<:BatchedCuMatrix{T,D1,D2}}) where {T,D1,D2}
    return TraceMatrix{T,D1,D2}  # BatchedCuMatrix's (D1,D2) become TraceMatrix's (D_M,D_N)
end
function trace_element_type(::Type{<:SharedCuMatrix{T,D1,D2}}) where {T,D1,D2}
    return TraceMatrix{T,D1,D2}
end
function trace_element_type(::Type{<:BatchedCuVector{T,D}}) where {T,D}
    return TraceVector{T,D}
end
function trace_element_type(::Type{<:SharedCuVector{T,D}}) where {T,D}
    return TraceVector{T,D}
end
function trace_element_type(::Type{<:BatchedCuScalar{T}}) where {T}
    return TraceScalar{T}
end
trace_element_type(::Type{SharedValue{T}}) where {T} = T

@generated function trace_element_type(::Type{BS}) where {T,C,BS<:BatchedStruct{T,C}}
    comp_types = C.parameters[2].parameters
    if T <: Tuple
        traced = Type[trace_element_type(ct) for ct in comp_types]
        result = Tuple{traced...}
        return :($result)
    end

    fnames = fieldnames(T)
    traced_by_field = Dict{Symbol,Type}()
    for (name, ct) in zip(C.parameters[1], comp_types)
        traced_by_field[name] = trace_element_type(ct)
    end

    base = Base.typename(T).wrapper
    new_params = Any[]
    for param in T.parameters
        replaced = false
        for f in fnames
            if fieldtype(T, f) === param && haskey(traced_by_field, f)
                push!(new_params, traced_by_field[f])
                replaced = true
                break
            end
        end
        replaced || push!(new_params, param)
    end
    result = base{new_params...}
    return :($result)
end

# -----------------------------------------------------------------------------
# Cholesky.{L,U} for trace-backed Cholesky
# -----------------------------------------------------------------------------
#
# Our `cholesky(::TraceMatrix)` overload stores the U factor and tags `uplo`
# as 'U'. Stdlib's `getproperty(::Cholesky, :L)` for that case does
# `LowerTriangular(copy(Cfactors'))`, and `copy(::Adjoint{T,<:AbstractMatrix})`
# materialises via scalar indexing — forbidden on `TraceMatrix`. Overriding
# `Base.copy` for the trace types would work but is a sharp edge: future code
# may rely on copy producing an independent object. Overriding the property
# accessor on `Cholesky{T,<:TraceMatrix}` is a narrower fix.

function Base.getproperty(C::Cholesky{T,<:TraceMatrix}, d::Symbol) where {T}
    if d === :L
        return LowerTriangular(getfield(C, :factors)')
    elseif d === :U
        return UpperTriangular(getfield(C, :factors))
    elseif d === :UL
        return Symmetric(getfield(C, :factors), :U)
    end
    return getfield(C, d)
end

# -----------------------------------------------------------------------------
# Wrapper capture
# -----------------------------------------------------------------------------
#
# `register_wrapped!` walks a stdlib wrapper chain over a phantom TraceMatrix
# from outermost to innermost, emitting a NewNode at each level. Called by
# primitives that consume wrapped trace values, so the tape captures the
# wrapper structure at the call site rather than at the wrapping site.

register_wrapped!(tape::Tape, M::TraceMatrix) = M.ref

function register_wrapped!(tape::Tape, A::Adjoint{T,S}) where {T,S}
    inner = register_wrapped!(tape, A.parent)
    return emit_new!(tape, Adjoint{T,S}, :parent, inner)
end

function register_wrapped!(tape::Tape, U::UpperTriangular{T,S}) where {T,S}
    inner = register_wrapped!(tape, U.data)
    return emit_new!(tape, UpperTriangular{T,S}, :data, inner)
end

function register_wrapped!(tape::Tape, L::LowerTriangular{T,S}) where {T,S}
    inner = register_wrapped!(tape, L.data)
    return emit_new!(tape, LowerTriangular{T,S}, :data, inner)
end

function register_wrapped!(tape::Tape, S::Symmetric{T,M}) where {T,M}
    inner = register_wrapped!(tape, S.data)
    return emit_new!(tape, Symmetric{T,M}, :data, inner)
end

# -----------------------------------------------------------------------------
# IAddSubWrapped — tag type for the `aI + bM` getter wrapper
# -----------------------------------------------------------------------------
#
# Used as the `NewNode.T` tag for an `aI + bM` view. The `±sI ± M` overloads
# eagerly emit a 3-field NewNode (`:parent => M.ref, :a => ConstNode,
# :b => ConstNode`) and return a `TraceMatrix{T,D_M,D_M}` whose `.ref` points
# at it — so downstream code dispatches through the ordinary
# `*(::TraceMatrix, ::TraceMatrix)` (and any other consumer) without
# wrapper-specific overloads. `arg_kernel_expr` lowers the NewNode to
# `IAddSubGetterMatrix(parent_view, a, b)`, folding the I-shift into the
# consuming sub-kernel for free.
#
# The struct is empty: it exists only so we can spell the type tag
# `IAddSubWrapped{T,D_M}` in `NewNode.T`, `arg_kernel_expr`, and `shape`.
# Instances are never constructed.

struct IAddSubWrapped{T,D_M} end

function _emit_iaddsub_wrap(M::TraceMatrix{T,D_M,D_M}, a::T, b::T) where {T,D_M}
    tape = M.tape
    aref = emit_const!(tape, a)
    bref = emit_const!(tape, b)
    fields = Pair{Symbol,NodeRef}[:parent => M.ref, :a => aref, :b => bref]
    lc = meta_at(tape, M.ref).lifecycle
    ref = push_node!(
        tape,
        NewNode(IAddSubWrapped{T,D_M}, fields),
        NodeMeta(IAddSubWrapped{T,D_M}, lc),
    )
    return TraceMatrix{T,D_M,D_M}(tape, ref)
end

# -----------------------------------------------------------------------------
# In-place primitive registry
# -----------------------------------------------------------------------------
#
# Two registries with different semantics:
#
# `inplace_arg(fn, types...)` — *semantic mutation*. Returns the 1-based index
# of the operand explicitly overwritten by a mutating scalar call (`cholesky!`,
# `ldiv!`). The planner must alias the result's slot to that operand's slot
# whether or not the operand is dead afterwards; failing to alias would change
# scalar Julia semantics.
#
# `inplace_safe_args(fn, types...)` — *optimisation opportunity*. Returns a
# tuple of 1-based operand positions whose slot the sub-kernel may overwrite
# *without* a read-after-write hazard. Auto in-place (planner pass) is allowed
# only when the operand at one of these positions is dead at the current node;
# the alias is purely a slot-reuse optimisation and the operand value is gone
# after the op either way.
#
# Whether `*(A, B)` could in-place over A or B depends on the sub-kernel, not
# the operator — matmul iterates A over multiple columns, so writing C[i,d]
# before reading A[i,d+1] would race across the warp. Hence the per-operator,
# per-arg-types registry: declarations live with the trace overloads.

inplace_arg(::typeof(cholesky!), ::Type{<:TraceMatrix}) = 1
inplace_arg(::typeof(ldiv!), ::Type{<:LowerTriangular}, ::Type{<:TraceMatrix}) = 2
inplace_arg(::typeof(ldiv!), ::Type{<:UpperTriangular}, ::Type{<:TraceMatrix}) = 2

# Default: no operand is alias-safe.
inplace_safe_args(::Any, ::Type...) = ()

# `+` / `-` on matrices and vectors: thread d operates entirely on column d
# (matrix) or element d (vector); reading and writing the same slot location
# within a single thread is fine.
inplace_safe_args(::typeof(+), ::Type{<:TraceMatrix}, ::Type{<:TraceMatrix}) = (1, 2)
inplace_safe_args(::typeof(-), ::Type{<:TraceMatrix}, ::Type{<:TraceMatrix}) = (1, 2)
inplace_safe_args(::typeof(+), ::Type{<:TraceVector}, ::Type{<:TraceVector}) = (1, 2)
inplace_safe_args(::typeof(-), ::Type{<:TraceVector}, ::Type{<:TraceVector}) = (1, 2)

# Non-mutating `cholesky(A) -> U`: each thread d touches only column d of both
# A and U, reading A[i,d] then writing U[i,d] within the same iteration.
inplace_safe_args(::typeof(cholesky), ::Type{<:TraceMatrix}) = (1,)

# Triangular solve: thread d copies RHS column d into a register vector, then
# writes the result column. The LHS (triangular factor) is read across the
# whole sweep and must not be aliased. Only the RHS (arg 2) is alias-safe.
inplace_safe_args(::typeof(\), ::Type{<:LowerTriangular}, ::Type{<:TraceMatrix}) = (2,)
inplace_safe_args(::typeof(\), ::Type{<:UpperTriangular}, ::Type{<:TraceMatrix}) = (2,)
inplace_safe_args(::typeof(\), ::Type{<:LowerTriangular}, ::Type{<:TraceVector}) = (2,)

# =============================================================================
# Trace entry point
# =============================================================================

abstract type InputSpec end

struct LeafInput <: InputSpec
    trace_type::Type
    lifecycle::Lifecycle
end

struct LiteralInput <: InputSpec
    val::Any
end

struct CompositeInput <: InputSpec
    T::Type
    fields::Vector{Pair{Symbol,InputSpec}}
end

function input_spec(x::BatchedCuMatrix)
    return LeafInput(trace_element_type(typeof(x)), BATCHED)
end
function input_spec(x::SharedCuMatrix)
    return LeafInput(trace_element_type(typeof(x)), SHARED)
end
function input_spec(x::BatchedCuVector)
    return LeafInput(trace_element_type(typeof(x)), BATCHED)
end
function input_spec(x::SharedCuVector)
    return LeafInput(trace_element_type(typeof(x)), SHARED)
end
function input_spec(x::SharedValue)
    return LiteralInput(x.value)
end
function input_spec(x::BatchedStruct{T}) where {T}
    comps = getfield(x, :components)
    trace_T = trace_element_type(typeof(x))
    fields = Pair{Symbol,InputSpec}[
        name => input_spec(comps[name]) for name in keys(comps)
    ]
    return CompositeInput(trace_T, fields)
end

input_trace_type(spec::LeafInput) = spec.trace_type
input_trace_type(spec::LiteralInput) = typeof(spec.val)
input_trace_type(spec::CompositeInput) = spec.T

input_cache_key(spec::LeafInput) = (:leaf, spec.trace_type, spec.lifecycle)
input_cache_key(spec::LiteralInput) = (:literal, typeof(spec.val), spec.val)
function input_cache_key(spec::CompositeInput)
    return (
        :composite,
        spec.T,
        Tuple(name => input_cache_key(child) for (name, child) in spec.fields),
    )
end

"""
    trace(f, input_specs) -> Tape

Build a tape by literally calling `f` with phantom inputs reconstructed from
structural input specs. Matrix leaves become InputNodes; literals become scalar
values; composites are rebuilt from their traced fields.
"""
function trace(f, input_specs::Vector{<:InputSpec})
    tape = Tape()
    phantoms = Any[_reconstruct_trace_arg!(tape, spec) for spec in input_specs]
    result = f(phantoms...)
    out_ref = result_to_ref!(tape, result)
    tape.output = out_ref
    return tape
end

function _reconstruct_trace_arg!(tape::Tape, spec::LeafInput)
    ref = push_node!(
        tape,
        InputNode(length(tape.inputs) + 1),
        NodeMeta(spec.trace_type, spec.lifecycle),
    )
    push!(tape.inputs, ref)
    (spec.trace_type <: TraceMatrix || spec.trace_type <: TraceVector) ||
        error("trace: leaf input type $(spec.trace_type) not supported")
    return spec.trace_type(tape, ref)
end

_reconstruct_trace_arg!(::Tape, spec::LiteralInput) = spec.val

function _reconstruct_trace_arg!(tape::Tape, spec::CompositeInput)
    vals = Any[_reconstruct_trace_arg!(tape, child) for (_, child) in spec.fields]
    if spec.T <: Tuple
        return tuple(vals...)
    end
    return spec.T(vals...)
end

# Convert the user function's return value into a tape output ref.
result_to_ref!(tape::Tape, M::TraceMatrix) = M.ref
result_to_ref!(tape::Tape, v::TraceVector) = v.ref
result_to_ref!(::Tape, s::TraceScalar) = s.ref
function result_to_ref!(tape::Tape, x::Union{Number,AbstractChar,Bool,Nothing})
    return emit_const!(tape, x)
end

function result_to_ref!(tape::Tape, t::Tuple)
    elem_refs = NodeRef[result_to_ref!(tape, e) for e in t]
    elem_types = Type[meta_at(tape, r).type for r in elem_refs]
    T = Tuple{elem_types...}
    names = Symbol[Symbol("_", k) for k in 1:length(elem_refs)]
    fields = Pair{Symbol,NodeRef}[n => r for (n, r) in zip(names, elem_refs)]
    lc = combined_lifecycle(tape, elem_refs)
    return push_node!(tape, NewNode(T, fields), NodeMeta(T, lc))
end

function result_to_ref!(tape::Tape, x)
    T = typeof(x)
    fields = fieldnames(T)
    isempty(fields) && error("result_to_ref!: unsupported output value of type $T")
    child_refs = NodeRef[result_to_ref!(tape, getfield(x, f)) for f in fields]
    pairs = Pair{Symbol,NodeRef}[f => r for (f, r) in zip(fields, child_refs)]
    lc = combined_lifecycle(tape, child_refs)
    return push_node!(tape, NewNode(T, pairs), NodeMeta(T, lc))
end
