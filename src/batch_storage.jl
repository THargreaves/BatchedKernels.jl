export allocate_batch, check_batch_copy

"""
    allocate_batch(prototype, n)

Allocate independent, writable storage for `n` members with the prototype's field
structure, scalar types and inner dimensions. Values are uninitialised. Numerical
shared fields become batched storage, so different writes can retain different
values. Non-numerical isbits `SharedValue` literals remain shared and must match on writes.
Composite types follow their allocated leaves, as with batch gathering.

Storage stays on the prototype's device (or host for host-backed batches). A batch
containing only shared numerical scalars defaults to CUDA storage. Mixed host/device or
multi-device numerical leaves are rejected. Dense arrays and regular range views
are supported, as for batch member assignment. No GPU fusion is involved.
"""
function allocate_batch(prototype::BatchedOrShared, n::Integer)
    0 <= n <= typemax(Int) || throw(ArgumentError("invalid batch capacity"))
    roots = _batch_roots(prototype)
    template = _batch_template(prototype)
    foreach(r -> _check_batch_device(template, r), roots)
    return _allocate_batch(prototype, Int(n), template)
end

function _batch_template(
    x::Union{BatchedCuVector,BatchedCuMatrix,BatchedCuScalar,SharedCuVector,SharedCuMatrix}
)
    return _member_storage(x.data)
end
_batch_template(x::Union{SharedScalar,SharedValue}) = nothing
_batch_template(x::AbstractVector) = _member_storage(x)
function _batch_template(x::BatchedStruct)
    return _first_batch_template(map(_batch_template, values(x.components)))
end
_first_batch_template(::Tuple{}) = nothing
function _first_batch_template(xs::Tuple)
    return first(xs) === nothing ? _first_batch_template(Base.tail(xs)) : first(xs)
end

function _batch_roots(
    x::Union{BatchedCuVector,BatchedCuMatrix,BatchedCuScalar,SharedCuVector,SharedCuMatrix}
)
    return [_member_storage(x.data)]
end
_batch_roots(x::Union{SharedScalar,SharedValue}) = ()
_batch_roots(x::AbstractVector) = [_member_storage(x)]
function _batch_roots(x::BatchedStruct)
    return Any[r for c in values(x.components) for r in _batch_roots(c)]
end

function _check_batch_device(a, b)
    (a isa CuArray) == (b isa CuArray) ||
        throw(ArgumentError("batch storage does not transfer between host and device"))
    if a isa CuArray
        CUDA.device(a) == CUDA.device(b) ||
            throw(ArgumentError("batch storage requires the same CUDA device"))
    end
    return nothing
end

function _allocate_batch(
    x::Union{BatchedCuVector{T,D},SharedCuVector{T,D}}, n, template
) where {T,D}
    data = similar(template, T, (D, n))
    return BatchedCuVector{T,D,typeof(data),_entry_view_type(typeof(data))}(data)
end
function _allocate_batch(
    x::Union{BatchedCuMatrix{T,D1,D2},SharedCuMatrix{T,D1,D2}}, n, template
) where {T,D1,D2}
    data = similar(template, T, (D1, D2, n))
    return BatchedCuMatrix{T,D1,D2,typeof(data),_entry_view_type(typeof(data))}(data)
end
function _allocate_batch(x::BatchedCuScalar, n, template)
    return BatchedCuScalar(similar(template, eltype(x), n))
end
function _allocate_batch(x::SharedValue{T}, n, template) where {T}
    isbitstype(T) || throw(ArgumentError("shared storage literals must be isbits"))
    T <: Number || return SharedValue(x.value, n)
    data = template === nothing ? CuArray{T}(undef, n) : similar(template, T, n)
    return BatchedCuScalar(data)
end
function _allocate_batch(x::SharedScalar{T}, n, template) where {T}
    data = template === nothing ? CuArray{T}(undef, n) : similar(template, T, n)
    return BatchedCuScalar(data)
end
_allocate_batch(x::AbstractVector, n, template) = similar(template, eltype(x), n)
function _allocate_batch(x::BatchedStruct{T}, n, template) where {T}
    components = map(c -> _allocate_batch(c, n, template), x.components)
    R = _composite_eltype(T, typeof(components))
    return BatchedStruct{R,typeof(components)}(components, n)
end

"""
    destination[indices::AbstractVector{<:Integer}] = source
    setindex!(destination, source, indices)

Copy all source members into distinct destination slots. Both batches must have
matching fields, scalar types and inner shapes; shared numerical sources can fill
batched destinations. Other slots are unchanged. Host or device integer indices
are accepted. Bounds, uniqueness, structure, device and non-aliasing are checked
before any write. Aliasing between destination fields or source/destination leaves is rejected;
gather a snapshot first when needed. Device execution errors do not roll back writes.
This bulk method requires matching scalar types, without implicit conversion.
`setindex!` returns `destination`; the assignment expression returns `source`, as
usual in Julia. Single-member vector assignment retains its existing semantics.
"""
function Base.setindex!(
    dest::BatchedOrShared, src::BatchedOrShared, indices::AbstractVector{<:Integer}
)
    eltype(indices) === Bool && throw(ArgumentError("logical indices are unsupported"))
    length(indices) == length(src) ||
        throw(DimensionMismatch("one destination per source member is required"))
    Base.require_one_based_indexing(indices)
    if indices isa CUDA.AnyCuArray
        foreach(r -> _check_batch_device(_member_storage(indices), r), _batch_roots(dest))
    end
    checkbounds(dest, indices)
    if length(indices) > 1 &&
        !all(view(indices, 2:length(indices)) .> view(indices, 1:(length(indices) - 1)))
        all(diff(sort(indices)) .> 0) ||
            throw(ArgumentError("destination indices must be distinct"))
    end
    index_storage = if indices isa Union{Array,CuArray,SubArray,Base.ReshapedArray}
        _member_storage(indices)
    else
        indices
    end
    any(r -> Base.mightalias(r, index_storage), _batch_roots(dest)) &&
        throw(ArgumentError("destination and index storage must not alias"))
    check_batch_copy(dest, src)
    _copy_batch!(dest, indices, src)
    return dest
end

"""
    check_batch_copy(destination, source)

Validate field structure, scalar types, inner shapes, structural literals, device
and non-aliasing for bulk `setindex!`, without reading uninitialised destination values
or writing storage. Batch lengths may differ. Throws on incompatibility.
"""
function check_batch_copy(dest::BatchedOrShared, src::BatchedOrShared)
    _check_batch_copy(dest, src)
    dstroots, srcroots = _batch_roots(dest), _batch_roots(src)
    for i in eachindex(dstroots), j in 1:(i - 1)
        Base.mightalias(dstroots[i], dstroots[j]) &&
            throw(ArgumentError("destination batch fields must not alias"))
    end
    for d in dstroots, s in srcroots
        _check_batch_device(d, s)
        Base.mightalias(d, s) && throw(
            ArgumentError(
                "batch source and destination must not alias; gather a snapshot first"
            ),
        )
    end
    return nothing
end

function _check_batch_copy(dest::BatchedStruct{T}, src::BatchedStruct{S}) where {T,S}
    Base.typename(T).wrapper === Base.typename(S).wrapper &&
    keys(dest.components) == keys(src.components) ||
        throw(ArgumentError("batch composite structures differ"))
    _composite_eltype(S, typeof(dest.components)) === T ||
        throw(ArgumentError("batch composite type parameters differ"))
    foreach(_check_batch_copy, values(dest.components), values(src.components))
    return nothing
end
function _check_batch_copy(
    dest::BatchedCuVector, src::Union{BatchedCuVector,SharedCuVector}
)
    return _check_batch_leaf(dest.data, src.data, size(dest.data)[1:1], size(src.data)[1:1])
end
function _check_batch_copy(
    dest::BatchedCuMatrix, src::Union{BatchedCuMatrix,SharedCuMatrix}
)
    return _check_batch_leaf(dest.data, src.data, size(dest.data)[1:2], size(src.data)[1:2])
end
function _check_batch_leaf(dest, src, ds, ss)
    ds == ss || throw(DimensionMismatch("batch member shapes differ"))
    eltype(dest) === eltype(src) || throw(ArgumentError("batch scalar types differ"))
    return nothing
end
function _check_batch_copy(dest::BatchedCuScalar, src::BatchedCuScalar)
    return _check_batch_leaf(dest.data, src.data, (), ())
end
function _check_batch_copy(
    dest::BatchedCuScalar{T}, src::Union{SharedScalar{S},SharedValue{S}}
) where {T,S}
    T === S && T <: Number || throw(ArgumentError("batch scalar types differ"))
    return nothing
end
function _check_batch_copy(dest::SharedScalar, src::SharedScalar)
    typeof(dest.value) === typeof(src.value) && isequal(dest.value, src.value) || throw(
        ArgumentError(
            "shared scalar destinations cannot change value; use allocate_batch for writable per-member storage",
        ),
    )
    return nothing
end
function _check_batch_copy(dest::SharedValue, src::SharedValue)
    typeof(dest.value) === typeof(src.value) && isequal(dest.value, src.value) ||
        throw(ArgumentError("shared structural literals must remain constant"))
    return nothing
end
function _check_batch_copy(dest::AbstractVector, src::AbstractVector)
    (dest isa BatchedOrShared || src isa BatchedOrShared) &&
        throw(ArgumentError("incompatible batch storage representations"))
    return _check_batch_leaf(dest, src, (), ())
end

function _copy_batch!(dest::BatchedStruct, indices, src::BatchedStruct)
    foreach(
        (d, s) -> _copy_batch!(d, indices, s),
        values(dest.components),
        values(src.components),
    )
    return dest
end
function _copy_batch!(
    dest::BatchedCuVector, indices, src::Union{BatchedCuVector,SharedCuVector}
)
    @views dest.data[:, indices] .= src.data
    return dest
end
function _copy_batch!(
    dest::BatchedCuMatrix, indices, src::Union{BatchedCuMatrix,SharedCuMatrix}
)
    @views dest.data[:, :, indices] .= src.data
    return dest
end
function _copy_batch!(dest::BatchedCuScalar, indices, src::BatchedCuScalar)
    dest.data[indices] = src.data
    return dest
end
function _copy_batch!(dest::BatchedCuScalar, indices, src::Union{SharedScalar,SharedValue})
    @views dest.data[indices] .= src.value
    return dest
end
_copy_batch!(dest::SharedValue, indices, src::SharedValue) = dest
_copy_batch!(dest::SharedScalar, indices, src::SharedScalar) = dest
function _copy_batch!(dest::AbstractVector, indices, src::AbstractVector)
    dest[indices] = src
    return dest
end
