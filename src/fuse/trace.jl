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
# Runtime-container → trace-element type map
# -----------------------------------------------------------------------------

function trace_element_type(::Type{<:BatchedCuMatrix{T,D1,D2}}) where {T,D1,D2}
    return TraceMatrix{T,D1,D2}  # BatchedCuMatrix's (D1,D2) become TraceMatrix's (D_M,D_N)
end
function trace_element_type(::Type{<:SharedCuMatrix{T,D1,D2}}) where {T,D1,D2}
    return TraceMatrix{T,D1,D2}
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
# In-place primitive registry
# -----------------------------------------------------------------------------
#
# A method here returns the 1-based index of the argument explicitly overwritten
# by a mutating scalar operation, so the planner can alias the result's slot to
# that argument's slot. Non-mutating scalar calls such as `cholesky(A)` and
# `L \ A` must not register here.

inplace_arg(::typeof(cholesky!), ::Type{<:TraceMatrix}) = 1
inplace_arg(::typeof(ldiv!), ::Type{<:LowerTriangular}, ::Type{<:TraceMatrix}) = 2
inplace_arg(::typeof(ldiv!), ::Type{<:UpperTriangular}, ::Type{<:TraceMatrix}) = 2

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
    spec.trace_type <: TraceMatrix ||
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
