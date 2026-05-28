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

export BatchedCuMatrix, BatchedCuVector
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
    V = typeof(view(data, :, :, 1))
    return BatchedCuMatrix{T,D1,D2,A,V}(data)
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
    V = typeof(view(data, :, 1))
    return BatchedCuVector{T,D,A,V}(data)
end
inner_shape(::Type{<:BatchedCuVector{T,D}}) where {T,D} = (D,)
inner_shape(x::BatchedCuVector) = inner_shape(typeof(x))
batch_size(x::BatchedCuVector) = size(x.data, 2)
Base.size(x::BatchedCuVector) = (batch_size(x),)
Base.IndexStyle(::Type{<:BatchedCuVector}) = IndexLinear()
Base.getindex(x::BatchedCuVector, i::Integer) = view(x.data, :, i)

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
# Shared union and helpers
# -----------------------------------------------------------------------------

const BatchedOrShared = Union{BatchedCuMatrix,BatchedCuVector,SharedCuMatrix,SharedCuVector}

is_shared_type(::Type{<:BatchedCuMatrix}) = false
is_shared_type(::Type{<:BatchedCuVector}) = false
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
