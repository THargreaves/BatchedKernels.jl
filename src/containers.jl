# =============================================================================
# Runtime container types
# =============================================================================
#
# Static-dim parameterisation: inner dimensions live in the type so the fuser
# can specialise. Batch axis is always 1D.
#
# - `BatchedCuMatrix` / `BatchedCuVector`: one element per batch entry, backed
#   by a single contiguous device array.
# - `SharedCuMatrix` / `SharedCuVector`: the same underlying array reused
#   across every batch entry; carries an explicit batch size so it satisfies
#   the `AbstractVector` contract.
# - `SharedValue`: a single scalar reused across the batch (used inside
#   composite values for fields such as `Cholesky.uplo`).
# - `BatchedStruct`: struct-of-arrays representation of a batch of composite
#   scalar values.

export BatchedCuMatrix, BatchedCuVector, BatchedCuScalar
export SharedCuMatrix, SharedCuVector
export SharedValue, BatchedStruct

# -----------------------------------------------------------------------------
# BatchedCuMatrix
# -----------------------------------------------------------------------------

struct BatchedCuMatrix{T,D1,D2,A<:AbstractArray{T,3},V} <: AbstractVector{V}
    data::A
end
function BatchedCuMatrix(data::A) where {T,A<:AbstractArray{T,3}}
    D1, D2 = size(data, 1), size(data, 2)
    return BatchedCuMatrix{T,D1,D2,A,_entry_view_type(A)}(data)
end
# Derive an entry's view type from the storage type, as Base does for the element
# type of an empty `map`, so empty batches need no (out-of-bounds) view.
function _entry_view_type(::Type{A}) where {A<:AbstractArray{<:Any,3}}
    return Base.promote_op(view, A, Colon, Colon, Int)
end
function _entry_view_type(::Type{A}) where {A<:AbstractArray{<:Any,2}}
    return Base.promote_op(view, A, Colon, Int)
end
inner_shape(::Type{<:BatchedCuMatrix{T,D1,D2}}) where {T,D1,D2} = (D1, D2)
inner_shape(x::BatchedCuMatrix) = inner_shape(typeof(x))
batch_size(x::BatchedCuMatrix) = size(x.data, 3)
Base.size(x::BatchedCuMatrix) = (batch_size(x),)
Base.IndexStyle(::Type{<:BatchedCuMatrix}) = IndexLinear()
Base.getindex(x::BatchedCuMatrix, i::Integer) = view(x.data, :, :, i)

# -----------------------------------------------------------------------------
# BatchedCuVector
# -----------------------------------------------------------------------------

struct BatchedCuVector{T,D,A<:AbstractArray{T,2},V} <: AbstractVector{V}
    data::A
end
function BatchedCuVector(data::A) where {T,A<:AbstractArray{T,2}}
    D = size(data, 1)
    return BatchedCuVector{T,D,A,_entry_view_type(A)}(data)
end
inner_shape(::Type{<:BatchedCuVector{T,D}}) where {T,D} = (D,)
inner_shape(x::BatchedCuVector) = inner_shape(typeof(x))
batch_size(x::BatchedCuVector) = size(x.data, 2)
Base.size(x::BatchedCuVector) = (batch_size(x),)
Base.IndexStyle(::Type{<:BatchedCuVector}) = IndexLinear()
Base.getindex(x::BatchedCuVector, i::Integer) = view(x.data, :, i)

# Member assignment is slice assignment: retain the packed storage and copy the
# values, as for eachcol(A)[i] = value. Restrict destinations to storage maps whose
# members cannot overlap (in particular, reject repeated-index views).
_member_storage(a::Union{Array,CuArray}) = a
function _member_storage(a::SubArray)
    all(i -> i isa Union{Integer,Base.Slice,AbstractRange}, parentindices(a)) ||
        throw(ArgumentError("member assignment requires regular range views"))
    any(i -> i isa AbstractRange && length(i) > 1 && iszero(step(i)), parentindices(a)) &&
        throw(ArgumentError("member assignment requires distinct destination indices"))
    return _member_storage(parent(a))
end
_member_storage(a::Base.ReshapedArray) = _member_storage(parent(a))
function _member_storage(a)
    return throw(ArgumentError("unsupported member assignment storage $(typeof(a))"))
end

"""
    batch[i::Integer] = value::AbstractVector

Copy `value` into member `i` of a `BatchedCuVector`, retaining its storage. The
source must have exactly the member's axes; scalar expansion is not performed.
Element conversion follows the backing array's copy operation, without rollback
if conversion or device execution fails. `setindex!` returns `batch`.

Dense host/CUDA arrays and regular range views are supported. Source and destination
must both be host arrays or reside on the same CUDA device. Overlapping sources
are snapshotted before writing. Existing views of the destination observe the
write; a non-aliasing source and other members are unchanged. This is an eager
operation, not mutation inside `fuse`.
"""
function Base.setindex!(x::BatchedCuVector, value::AbstractVector, i::Integer)
    checkbounds(x, i)
    dest = x[i]
    axes(dest) == axes(value) || throw(DimensionMismatch("batch member axes differ"))
    dstroot, srcroot = _member_storage(x.data), _member_storage(value)
    (dstroot isa CuArray) == (srcroot isa CuArray) ||
        throw(ArgumentError("member assignment does not transfer between host and device"))
    if dstroot isa CuArray
        CUDA.device(dstroot) == CUDA.device(srcroot) ||
            throw(ArgumentError("member assignment requires the same CUDA device"))
    end
    # Comparing a dense CUDA view to a SubArray can miss overlap. Compare their
    # unwrapped storage instead; conservative positives only cost one member copy.
    source = Base.mightalias(dstroot, srcroot) ? copy(value) : value
    copyto!(dest, source)
    return x
end

# -----------------------------------------------------------------------------
# BatchedCuScalar
# -----------------------------------------------------------------------------
#
# One scalar per batch entry, backed by a length-N device vector. Reductions
# produce these containers; they can also feed subsequent fused calls.

struct BatchedCuScalar{T,A<:AbstractVector{T}} <: AbstractVector{T}
    data::A
end
batch_size(x::BatchedCuScalar) = length(x.data)
Base.size(x::BatchedCuScalar) = size(x.data)
Base.IndexStyle(::Type{<:BatchedCuScalar}) = IndexLinear()
Base.getindex(x::BatchedCuScalar, i::Integer) = x.data[i]

# -----------------------------------------------------------------------------
# SharedCuMatrix
# -----------------------------------------------------------------------------

struct SharedCuMatrix{T,D1,D2,A<:AbstractMatrix{T}} <: AbstractVector{A}
    data::A
    batchsize::Int
end
function SharedCuMatrix(data::A, N::Int) where {T,A<:AbstractMatrix{T}}
    D1, D2 = size(data, 1), size(data, 2)
    return SharedCuMatrix{T,D1,D2,A}(data, N)
end
inner_shape(::Type{<:SharedCuMatrix{T,D1,D2}}) where {T,D1,D2} = (D1, D2)
inner_shape(x::SharedCuMatrix) = inner_shape(typeof(x))
batch_size(x::SharedCuMatrix) = x.batchsize
Base.size(x::SharedCuMatrix) = (batch_size(x),)
Base.IndexStyle(::Type{<:SharedCuMatrix}) = IndexLinear()
Base.getindex(x::SharedCuMatrix, ::Integer) = x.data

# -----------------------------------------------------------------------------
# SharedCuVector
# -----------------------------------------------------------------------------

struct SharedCuVector{T,D,A<:AbstractVector{T}} <: AbstractVector{A}
    data::A
    batchsize::Int
end
function SharedCuVector(data::A, N::Int) where {T,A<:AbstractVector{T}}
    D = length(data)
    return SharedCuVector{T,D,A}(data, N)
end
inner_shape(::Type{<:SharedCuVector{T,D}}) where {T,D} = (D,)
inner_shape(x::SharedCuVector) = inner_shape(typeof(x))
batch_size(x::SharedCuVector) = x.batchsize
Base.size(x::SharedCuVector) = (batch_size(x),)
Base.IndexStyle(::Type{<:SharedCuVector}) = IndexLinear()
Base.getindex(x::SharedCuVector, ::Integer) = x.data

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

is_shared_type(::Type{<:BatchedCuMatrix}) = false
is_shared_type(::Type{<:BatchedCuVector}) = false
is_shared_type(::Type{<:BatchedCuScalar}) = false
is_shared_type(::Type{<:SharedCuMatrix}) = true
is_shared_type(::Type{<:SharedCuVector}) = true

# -----------------------------------------------------------------------------
# SharedValue
# -----------------------------------------------------------------------------

struct SharedValue{T} <: AbstractVector{T}
    value::T
    batch_n::Int
end
Base.eltype(::Type{SharedValue{T}}) where {T} = T
Base.size(x::SharedValue) = (x.batch_n,)
Base.length(x::SharedValue) = x.batch_n
Base.IndexStyle(::Type{<:SharedValue}) = IndexLinear()
Base.getindex(x::SharedValue, ::Integer) = x.value
batch_size(x::SharedValue) = x.batch_n

# -----------------------------------------------------------------------------
# BatchedStruct
# -----------------------------------------------------------------------------

struct BatchedStruct{T,C<:NamedTuple} <: AbstractVector{T}
    components::C
    batch_n::Int
end
"""
    BatchedStruct(T, components::NamedTuple)

Build a batch of composite values of type `T` from named batches of its fields.
`T` may be a parametric struct type whose unresolved parameters appear directly
as field types. Those parameters are filled from the component element types,
without indexing or copying their storage. Fields sharing a parameter must have
identical element types; parameter bounds remain enforced.
Components must have the declared field names and equal lengths. Named fields
are reordered to declaration order, and integer fields retain their exact types.

A concrete `T` preserves its declared field types, including abstract fields.
Supply a concrete type for nested, value or unused parameters that cannot be
obtained from a direct field type. No constructor inference, type promotion or
custom constructor computation is performed. At least one field is required
so the batch length can be inferred.
"""
function BatchedStruct(::Type{T}, components::NamedTuple) where {T}
    body = Base.unwrap_unionall(T)
    isstructtype(body) && !(body <: Tuple) ||
        throw(ArgumentError("BatchedStruct requires a composite struct type"))
    names = fieldnames(body)
    isempty(names) &&
        throw(ArgumentError("cannot infer batch length for a fieldless struct"))
    length(components) == length(names) && all(name -> haskey(components, name), names) ||
        throw(ArgumentError("components must have the fields $names"))
    ordered = NamedTuple{names}(components)
    all(c -> c isa AbstractVector, values(ordered)) ||
        throw(ArgumentError("each composite component must be a batch vector"))
    n = length(first(ordered))
    all(c -> length(c) == n, values(ordered)) ||
        throw(DimensionMismatch("composite component batch lengths differ"))
    foreach(Base.require_one_based_indexing, values(ordered))
    types = map(eltype, values(ordered))
    R = _composite_constructor_type(T, Tuple{types...})
    isconcretetype(R) && R <: T || throw(
        ArgumentError(
            "cannot determine one concrete element type for $T; specify it explicitly"
        ),
    )
    all(type <: fieldtype(R, name) for (name, type) in zip(names, types)) ||
        throw(ArgumentError("component element types do not match the fields of $R"))
    return BatchedStruct{R,typeof(ordered)}(ordered, n)
end

# This is declared-type substitution, not inference of a constructor's return
# type. Generate only from type metadata so the chosen element type is a literal.
@generated function _composite_constructor_type(
    ::Type{T}, ::Type{Types}
) where {T,Types<:Tuple}
    isconcretetype(T) && return :($T)
    body = Base.unwrap_unionall(T)
    params = Any[body.parameters...]
    fields = fieldtypes(body)
    for (i, parameter) in enumerate(params)
        parameter isa TypeVar || continue
        positions = findall(field -> field === parameter, fields)
        if isempty(positions)
            message = "Parameter $(parameter.name) is not a direct field type of $T; supply a concrete type"
            return :(throw(ArgumentError($message)))
        end
        replacement = Types.parameters[first(positions)]
        if !all(j -> Types.parameters[j] === replacement, positions)
            message = "Fields sharing parameter $(parameter.name) need identical component element types"
            return :(throw(ArgumentError($message)))
        end
        params[i] = replacement
    end
    result = try
        Core.apply_type(Base.typename(body).wrapper, params...)
    catch err
        err isa TypeError || rethrow()
        return :(throw(
            ArgumentError("component element types violate declared parameter bounds")
        ))
    end
    return :($result)
end

Base.eltype(::Type{<:BatchedStruct{T}}) where {T} = T
Base.size(x::BatchedStruct) = (x.batch_n,)
Base.length(x::BatchedStruct) = x.batch_n
Base.IndexStyle(::Type{<:BatchedStruct}) = IndexLinear()
@generated function Base.getindex(x::BatchedStruct{T}, i::Integer) where {T}
    if T <: Tuple
        n = length(T.parameters)
        cs = [:(getfield(x, :components)[$k][i]) for k in 1:n]
        return :(tuple($(cs...)))
    else
        fields = fieldnames(T)
        cs = [:(getfield(x, :components)[$(QuoteNode(f))][i]) for f in fields]
        return :(T($(cs...)))
    end
end
function Base.show(io::IO, x::BatchedStruct{T}) where {T}
    return print(io, "BatchedStruct{", T, "} with ", x.batch_n, " elements")
end
function Base.show(io::IO, ::MIME"text/plain", x::BatchedStruct{T}) where {T}
    println(io, "BatchedStruct{", T, "} with ", x.batch_n, " elements:")
    for (k, v) in pairs(x.components)
        println(io, "  .", k, " :: ", typeof(v))
    end
end

# Rebuild a parametric struct from its declared field-to-parameter relationships.
# Concrete parameter values are not identities: e.g. H and R may both be Matrix
# while their distinct parameters must become differently shaped TraceMatrices.
function _replace_composite_field_types(::Type{T}, replacements) where {T}
    wrapper = Base.typename(T).wrapper
    body = wrapper
    variables = TypeVar[]
    while body isa UnionAll
        push!(variables, body.var)
        body = body.body
    end
    params = Any[T.parameters...]
    for (i, variable) in enumerate(variables)
        replacement = nothing
        for name in fieldnames(body)
            if fieldtype(body, name) === variable && haskey(replacements, name)
                candidate = replacements[name]
                if replacement !== nothing && replacement !== candidate
                    throw(
                        ArgumentError(
                            "Fields sharing type parameter $(variable.name) need matching traced types",
                        ),
                    )
                end
                replacement = candidate
            end
        end
        replacement === nothing || (params[i] = replacement)
    end
    return Core.apply_type(wrapper, params...)
end

# -----------------------------------------------------------------------------
# Batch gathering
# -----------------------------------------------------------------------------

const BatchedOrShared = Union{
    BatchedCuMatrix,
    BatchedCuVector,
    BatchedCuScalar,
    SharedCuMatrix,
    SharedCuVector,
    SharedValue,
    BatchedStruct,
}

"""
    batch[idxs::AbstractVector{<:Integer}]

Eagerly gather the entries `idxs` of a batch into a new batch of length
`length(idxs)`. Indices may repeat or be reordered, and every leaf of a composite
batch uses the same indices. Batched leaves are copied into independent storage,
retaining their scalar element types exactly; shared leaves and `SharedValue`s keep
their data with the new batch length.

Indices may reside on the host or on the device holding the batch storage; device
indices are bounds-checked on the device. Gathered storage is contiguous, so a leaf
backed by a strided view gathers with a different element (view) type. Composite
element types then follow their leaves, as for fused outputs: each declared field
type the gathered field still satisfies is kept, and the type parameters used
directly as the other field types are substituted. Composites whose types cannot be
relabelled this way, and logical indexing, raise an `ArgumentError`.
"""
function Base.getindex(x::BatchedOrShared, idxs::AbstractVector{<:Integer})
    eltype(idxs) === Bool &&
        throw(ArgumentError("logical indexing of batches is not supported"))
    # Check once for the whole composite: each device-index check is a reduction
    # followed by a synchronisation, so the leaves below skip their own checks.
    checkbounds(x, idxs)
    return _gather(x, idxs)
end

# The public constructors read inner dimensions from runtime sizes; reuse the
# input's static dimensions so gathering stays inferable.
function _gather(x::BatchedCuMatrix{T,D1,D2}, idxs) where {T,D1,D2}
    data = @inbounds x.data[:, :, idxs]
    return BatchedCuMatrix{T,D1,D2,typeof(data),_entry_view_type(typeof(data))}(data)
end
function _gather(x::BatchedCuVector{T,D}, idxs) where {T,D}
    data = @inbounds x.data[:, idxs]
    return BatchedCuVector{T,D,typeof(data),_entry_view_type(typeof(data))}(data)
end
_gather(x::BatchedCuScalar, idxs) = BatchedCuScalar(@inbounds x.data[idxs])
function _gather(x::SharedCuMatrix{T,D1,D2,A}, idxs) where {T,D1,D2,A}
    return SharedCuMatrix{T,D1,D2,A}(x.data, length(idxs))
end
function _gather(x::SharedCuVector{T,D,A}, idxs) where {T,D,A}
    return SharedCuVector{T,D,A}(x.data, length(idxs))
end
_gather(x::SharedValue, idxs) = SharedValue(x.value, length(idxs))
# Composites may also hold ordinary host or device vectors.
_gather(x::AbstractVector, idxs) = x[idxs]
function _gather(x::BatchedStruct{T}, idxs) where {T}
    components = map(c -> _gather(c, idxs), getfield(x, :components))
    R = _composite_eltype(T, typeof(components))
    return BatchedStruct{R,typeof(components)}(components, length(idxs))
end

# Derived composite batches (fused outputs, gathered batches) are labelled from
# their leaves. Declared field types the leaf element types satisfy are kept, so
# a gather retains its input type unless a view-backed leaf was copied.
function _leaf_composite_type(::Type{T}, ::Type{C}) where {T,C<:NamedTuple}
    types = map(eltype, fieldtypes(C))
    T <: Tuple && return Tuple{map((e, d) -> e<:d ? d : e, types, fieldtypes(T))...}
    names = fieldnames(C)
    replacements = Dict{Symbol,Any}(
        name => e for (name, e) in zip(names, types) if !(e <: fieldtype(T, name))
    )
    isempty(replacements) && return T
    R = try
        _replace_composite_field_types(T, replacements)
    catch err
        err isa TypeError || rethrow()
        nothing
    end
    if R === nothing || !all(e <: fieldtype(R, name) for (name, e) in zip(names, types))
        fields = join(sort!(collect(keys(replacements))), ", ")
        throw(
            ArgumentError(
                "cannot label $T from the element types of fields $fields; only type parameters used directly as field types can be substituted",
            ),
        )
    end
    return R
end

@generated function _composite_eltype(::Type{T}, ::Type{C}) where {T,C<:NamedTuple}
    result = try
        _leaf_composite_type(T, C)
    catch err
        err isa ArgumentError || rethrow()
        return :(throw($err))
    end
    return :($result)
end
