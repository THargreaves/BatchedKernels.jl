import Base: @propagate_inbounds
import LinearAlgebra: AdjOrTransAbsMat, wrapperop
using KernelAbstractions.Extras: @unroll
using StaticArrays: MVector

export DualAccessMatrix,
    SingleAccessMatrix,
    RegisterMatrix,
    Orientation,
    RowOriented,
    ColOriented,
    BothOriented,
    AccessConvention,
    RowAccess,
    ColAccess,
    orientation,
    flip,
    SharedMatrix,
    IAddSubSetterMatrix,
    IAddSubGetterMatrix,
    SharedVector,
    BatchedVector
export BlockMatrix_2_1, BlockMatrixLowerTrig_2_2
export intermediate_layout_load!, intermediate_layout_write!
export interm_to_dual_transfer!, dual_to_interm_transfer!
export single_to_register!, register_to_single!
export shared_matrix_load!, shared_vector_load!
export vector_load!, vector_write!
export scalar_stage!, scalar_write!
export single_region_elems, dual_region_elems

"""
Abstraction of shared memory layout for a matrix accessible both column and row-wise.
"""
struct DualAccessMatrix{T,D} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
    offset::Int32
end

"""
Constructor for DualAccessMatrix, for memory layout where one warp handles multiple matrices.
Meant for smaller matrices.
"""
function DualAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, warp_matrix_id::Int32
) where {T,D}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32

    n_mats_per_warp = 32i32 ÷ D
    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    offset = (wid - 1i32) * warp_shmem_size + warp_matrix_id - 1i32

    return DualAccessMatrix{T,D}(shmem, offset)
end
function DualAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, wid::Int32, warp_matrix_id::Int32
) where {T,D}
    n_mats_per_warp = 32i32 ÷ D
    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    offset = (wid - 1i32) * warp_shmem_size + warp_matrix_id - 1i32

    return DualAccessMatrix{T,D}(shmem, offset)
end

"""
Function for calculating the stride DualAccessMatrix memory layout where
one wrap handles multiple matrices.
"""
@inline function _compute_stride(::Val{D}) where {D}
    n_mats_per_warp = 32i32 ÷ D
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    return n_mats_per_warp * D + padding
end

"""
Function for calculating the number of matrices handled per warp, memory layout
where one warp handles multiple matrices.
"""
@inline function _compute_n_mats_per_warp(::Val{D}) where {D}
    return 32i32 ÷ D
end

"""
Get index method for memory layout where one warp handles multiple matrices
"""
Base.@propagate_inbounds @inline function Base.getindex(
    A::DualAccessMatrix{T,D}, i::Int32, j::Int32
) where {T,D}
    stride = _compute_stride(Val(D))
    n_mats_per_warp = _compute_n_mats_per_warp(Val(D))
    return A.shmem[A.offset + (j - 1i32) * stride + (i - 1i32) * n_mats_per_warp + 1i32]
end

"""
Set index method for memory layout where one warp handles multiple matrices
"""
Base.@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrix{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    stride = _compute_stride(Val(D))
    n_mats_per_warp = _compute_n_mats_per_warp(Val(D))
    return A.shmem[A.offset + (j - 1i32) * stride + (i - 1i32) * n_mats_per_warp + 1i32] = v
end

struct BlockMatrix_2_1{T,D,Mtop<:AbstractMatrix{T},Mbot<:AbstractMatrix{T}} <:
       AbstractMatrix{T}
    top::Mtop
    bot::Mbot
end

function BlockMatrix_2_1(
    top::Mtop, bot::Mbot, ::Val{D}, warp_matrix_id::Int32
) where {T,D,Mtop<:AbstractMatrix{T},Mbot<:AbstractMatrix{T}}
    return BlockMatrix_2_1{T,D,Mtop,Mbot}(top, bot)
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::BlockMatrix_2_1{T,D}, i::Int32, j::Int32
) where {T,D}
    if i <= D
        return A.top[i, j]
    else
        return A.bot[i - D, j]
    end
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::BlockMatrix_2_1{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    if i <= D
        return A.top[i, j] = v
    else
        return A.bot[i - D, j] = v
    end
end

Base.size(::BlockMatrix_2_1{T,D}) where {T,D} = (2i32 * D, D)

struct BlockMatrixLowerTrig_2_2{
    T,D,Mtop<:AbstractMatrix{T},Mbotleft<:AbstractMatrix{T},Mbotright<:AbstractMatrix{T}
} <: AbstractMatrix{T}
    top::Mtop
    bot_left::Mbotleft
    bot_right::Mbotright
end

function BlockMatrixLowerTrig_2_2(
    top::Mtop, bot_left::Mbotleft, bot_right::Mbotright, ::Val{D}, warp_matrix_id::Int32
) where {
    T,D,Mtop<:AbstractMatrix{T},Mbotleft<:AbstractMatrix{T},Mbotright<:AbstractMatrix{T}
}
    return BlockMatrixLowerTrig_2_2{T,D,Mtop,Mbotleft,Mbotright}(top, bot_left, bot_right)
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::BlockMatrixLowerTrig_2_2{T,D}, i::Int32, j::Int32
) where {T,D}
    if i <= D
        if j > D
            return zero(T)
        else
            return A.top[i, j]
        end
    else
        i -= D
        if j > D
            j -= D
            return A.bot_right[i, j]
        else
            return A.bot_left[i, j]
        end
    end
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::BlockMatrixLowerTrig_2_2{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    if i <= D
        if j > D
            return v
        else
            return A.top[i, j] = v
        end
    else
        i -= D
        if j > D
            j -= D
            return A.bot_right[i, j] = v
        else
            return A.bot_left[i, j] = v
        end
    end
end
Base.size(::BlockMatrixLowerTrig_2_2{T,D}) where {T,D} = (2i32 * D, 2i32 * D)

@propagate_inbounds @inline function Base.getindex(
    A::BlockMatrix_2_1{T,D}, i::Int, j::Int
) where {T,D}
    return getindex(A, Int32(i), Int32(j))
end

@propagate_inbounds @inline function Base.setindex!(
    A::BlockMatrix_2_1{T,D}, v::T, i::Int, j::Int
) where {T,D}
    return setindex!(A, v, Int32(i), Int32(j))
end

@propagate_inbounds @inline function Base.getindex(
    A::BlockMatrixLowerTrig_2_2{T,D}, i::Int, j::Int
) where {T,D}
    return getindex(A, Int32(i), Int32(j))
end

@propagate_inbounds @inline function Base.setindex!(
    A::BlockMatrixLowerTrig_2_2{T,D}, v::T, i::Int, j::Int
) where {T,D}
    return setindex!(A, v, Int32(i), Int32(j))
end

# Wrappers to handle Int32 case
@propagate_inbounds Base.getindex(A::AdjOrTransAbsMat{T}, i::Int32, j::Int32) where {T} =
    wrapperop(A)(A.parent[j, i])::T

@propagate_inbounds Base.setindex!(
    A::AdjOrTransAbsMat{T}, v, i::Int32, j::Int32
) where {T} = A.parent[j, i] = wrapperop(A)(convert(T, v))

# Support regular Int indexing (needed for Adjoint and other wrappers)
@propagate_inbounds @inline function Base.getindex(
    A::DualAccessMatrix{T,D}, i::Int, j::Int
) where {T,D}
    return getindex(A, Int32(i), Int32(j))
end

@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrix{T,D}, v::T, i::Int, j::Int
) where {T,D}
    return setindex!(A, v, Int32(i), Int32(j))
end

@inline Base.size(::DualAccessMatrix{T,D}) where {T,D} = (D, D)
@inline Base.length(::DualAccessMatrix{T,D}) where {T,D} = D * D
@inline Base.IndexStyle(::Type{<:DualAccessMatrix}) = IndexCartesian()

########################
#### BATCHED VECTOR ####
########################

"""
Abstraction of shared memory layout for a vector in a batch.
Vectors are stored contiguously without padding.
"""
struct BatchedVector{T,D} <: AbstractVector{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
    offset::Int32
end

function BatchedVector(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, warp_vector_id::Int32
) where {T,D}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32

    n_vecs_per_warp = 32i32 ÷ D
    offset = ((wid - 1i32) * n_vecs_per_warp + (warp_vector_id - 1i32)) * D

    return BatchedVector{T,D}(shmem, offset)
end

Base.@propagate_inbounds @inline function Base.getindex(
    v::BatchedVector{T,D}, i::Int32
) where {T,D}
    return v.shmem[v.offset + i]
end

Base.@propagate_inbounds @inline function Base.setindex!(
    v::BatchedVector{T,D}, val::T, i::Int32
) where {T,D}
    return v.shmem[v.offset + i] = val
end

@inline Base.size(::BatchedVector{T,D}) where {T,D} = (D,)
@inline Base.length(::BatchedVector{T,D}) where {T,D} = D
@inline Base.IndexStyle(::Type{<:BatchedVector}) = IndexLinear()

@inline function vector_load!(
    shmem, global_arr, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_vecs_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_vecs_per_block = n_warps * n_vecs_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    global_offset = (bid - 1i32) * n_vecs_per_block * D
    shmem_offset = (wid - 1i32) * n_vecs_per_warp * D

    if lid <= n_vecs_per_warp * D
        raw_idx = shmem_offset + lid
        raw_vec = div(raw_idx - 1i32, D) + 1i32
        grid_vec_load = raw_vec + (bid - 1i32) * n_vecs_per_block

        @inbounds if grid_vec_load <= N
            dest_idx = raw_idx
            src_idx = global_offset + raw_idx

            shmem[dest_idx] = global_arr[src_idx]
        end
    end

    return nothing
end

@inline function vector_load!(
    shmem, global_arr, ::Val{D1}, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D1,D,nthreads}
    n_vecs_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_vecs_per_block = n_warps * n_vecs_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    warp_shmem_elem = n_vecs_per_warp * D1

    global_offset = (bid - 1i32) * n_vecs_per_block * D1

    if lid <= n_vecs_per_warp * D1
        raw_idx = (wid - 1i32) * warp_shmem_elem + lid
        vec_idx = mod1(lid, D1)

        raw_vec = div(raw_idx - 1i32, D1) + 1i32
        grid_vec_load = raw_vec + (bid - 1i32) * n_vecs_per_block

        if grid_vec_load <= N
            dest_idx = (raw_vec - 1i32) * D + vec_idx
            src_idx = global_offset + raw_idx

            @inbounds shmem[dest_idx] = global_arr[src_idx]
        end
    end

    return nothing
end

@inline function vector_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_vecs_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_vecs_per_block = n_warps * n_vecs_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    global_offset = (bid - 1i32) * n_vecs_per_block * D
    shmem_offset = (wid - 1i32) * n_vecs_per_warp * D

    if lid <= n_vecs_per_warp * D
        raw_idx = shmem_offset + lid
        raw_vec = div(raw_idx - 1i32, D) + 1i32
        grid_vec_store = raw_vec + (bid - 1i32) * n_vecs_per_block

        @inbounds if grid_vec_store <= N
            src_idx = raw_idx
            dest_idx = global_offset + raw_idx

            global_arr[dest_idx] = shmem[src_idx]
        end
    end

    return nothing
end

@inline function vector_write!(
    global_arr, shmem, ::Val{D1}, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D1,D,nthreads}
    n_vecs_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_vecs_per_block = n_warps * n_vecs_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    warp_shmem_elem = n_vecs_per_warp * D1

    global_offset = (bid - 1i32) * n_vecs_per_block * D1

    if lid <= n_vecs_per_warp * D1
        raw_idx = (wid - 1i32) * warp_shmem_elem + lid
        vec_idx = mod1(lid, D1)

        raw_vec = div(raw_idx - 1i32, D1) + 1i32
        grid_vec_store = raw_vec + (bid - 1i32) * n_vecs_per_block

        if grid_vec_store <= N
            src_idx = (raw_vec - 1i32) * D + vec_idx
            dest_idx = global_offset + raw_idx

            @inbounds global_arr[dest_idx] = shmem[src_idx]
        end
    end

    return nothing
end

"""Physical ownership of matrix entries, independent of an operation's access convention."""
abstract type Orientation end
struct RowOriented <: Orientation end
struct ColOriented <: Orientation end
struct BothOriented <: Orientation end

"""Logical indexing convention selected statically by a compute operation."""
abstract type AccessConvention end
struct RowAccess <: AccessConvention end
struct ColAccess <: AccessConvention end

@inline flip(::RowOriented) = ColOriented()
@inline flip(::ColOriented) = RowOriented()
@inline flip(::BothOriented) = BothOriented()
@inline flip(::RowAccess) = ColAccess()
@inline flip(::ColAccess) = RowAccess()

@inline function _validate_compute_shape(::Val{M}, ::Val{N}, ::Val{D}) where {M,N,D}
    (M isa Integer && N isa Integer && D isa Integer && 1 <= M <= D <= 32 && 1 <= N <= D) ||
        throw(ArgumentError("compute layouts require 1 <= M,N <= D_MAX <= 32"))
    return nothing
end

@inline function _single_pad_interval(::Val{D}) where {D}
    d = Int32(D)
    return (32i32 ÷ (d & -d)) * d
end

# The last stored element determines the footprint; no trailing pad is necessary.
@inline function _single_warp_stride(::Val{D}) where {D}
    _validate_compute_shape(Val(D), Val(D), Val(D))
    d = Int32(D)
    q = (32i32 ÷ d) * d * d
    return q + (q - 1i32) ÷ _single_pad_interval(Val(D))
end

@inline function _validate_layout_threads(::Val{nthreads}) where {nthreads}
    (nthreads isa Integer && 32 <= nthreads <= 1024 && nthreads % 32 == 0) ||
        throw(ArgumentError("layout regions require 32..1024 threads in complete warps"))
    return nothing
end

"""Element count of a new padded single-compute region for a complete thread block."""
@inline function single_region_elems(::Val{D}, ::Val{nthreads}) where {D,nthreads}
    _validate_layout_threads(Val(nthreads))
    return _single_warp_stride(Val(D)) * (Int32(nthreads) ÷ 32i32)
end

@inline function _dual_warp_stride(::Val{D}) where {D}
    _validate_compute_shape(Val(D), Val(D), Val(D))
    d = Int32(D)
    n = 32i32 ÷ d
    padding = mod(n - mod(n * d, 32i32), 32i32)
    return n * d * d + padding * (d - 1i32)
end

"""Element count of an existing dual-layout region for a complete thread block."""
@inline function dual_region_elems(::Val{D}, ::Val{nthreads}) where {D,nthreads}
    _validate_layout_threads(Val(nthreads))
    return _dual_warp_stride(Val(D)) * (Int32(nthreads) ÷ 32i32)
end

@inline _single_raw_index(
    ::Val{D}, ::RowOriented, inner::Int32, i::Int32, j::Int32
) where {D} = inner + (j - 1i32) * Int32(D) + i - 1i32
@inline _single_raw_index(
    ::Val{D}, ::ColOriented, inner::Int32, i::Int32, j::Int32
) where {D} = inner + (i - 1i32) * Int32(D) + j - 1i32

# Offsets are zero-based; the returned backing-storage index is one-based.
@inline function _single_address(
    ::Val{D}, o, outer::Int32, inner::Int32, i::Int32, j::Int32
) where {D}
    r = _single_raw_index(Val(D), o, inner, i, j)
    return outer + r + r ÷ _single_pad_interval(Val(D)) + 1i32
end

@inline function _single_transfer_address(
    ::Val{D}, o, wid::Int32, mid::Int32, i::Int32, j::Int32
) where {D}
    return _single_address(
        Val(D),
        o,
        (wid - 1i32) * _single_warp_stride(Val(D)),
        (mid - 1i32) * Int32(D) * Int32(D),
        i,
        j,
    )
end

"""
    SingleAccessMatrix(storage, Val(M), Val(N), Val(D_MAX), orientation, wid, matrix_id)

A logical M×N matrix in a padded D_MAX×D_MAX compute tile. Row orientation stores
columns consecutively; col orientation transposes the physical map. `wid` and
`matrix_id` are one-based warp and within-warp matrix indices. Storage is concrete
(one-based shared device storage in kernels, or a one-based vector for CPU layout
validation).

This map is separate from the legacy packed rectangular intermediate transfers.
"""
struct SingleAccessMatrix{T,M,N,D,O<:Union{RowOriented,ColOriented},S<:AbstractVector{T}} <:
       AbstractMatrix{T}
    shmem::S
    outer_offset::Int32
    inner_offset::Int32

    @inline function SingleAccessMatrix(
        shmem::S, ::Val{M}, ::Val{N}, ::Val{D}, ::O, wid::Int32, matrix_id::Int32
    ) where {T,M,N,D,O<:Union{RowOriented,ColOriented},S<:AbstractVector{T}}
        _validate_compute_shape(Val(M), Val(N), Val(D))
        d = Int32(D)
        stride = _single_warp_stride(Val(D))
        @boundscheck begin
            Base.require_one_based_indexing(shmem)
            (wid >= 1i32 && 1i32 <= matrix_id <= 32i32 ÷ d) ||
                throw(ArgumentError("invalid warp or within-warp matrix index"))
            wid * stride <= length(shmem) || throw(BoundsError(shmem, wid * stride))
        end
        return new{T,M,N,D,O,S}(shmem, (wid - 1i32) * stride, (matrix_id - 1i32) * d * d)
    end
end

@inline Base.size(::SingleAccessMatrix{T,M,N}) where {T,M,N} = (Int(M), Int(N))
@inline Base.IndexStyle(::Type{<:SingleAccessMatrix}) = IndexCartesian()
@inline orientation(::Type{<:SingleAccessMatrix{T,M,N,D,O}}) where {T,M,N,D,O} = O()
@inline orientation(::Type{<:DualAccessMatrix}) = BothOriented()

@inline function _single_index(
    A::SingleAccessMatrix{T,M,N,D,O}, i::Int32, j::Int32
) where {T,M,N,D,O}
    return _single_address(Val(D), O(), A.outer_offset, A.inner_offset, i, j)
end

@propagate_inbounds @inline function Base.getindex(
    A::SingleAccessMatrix, i::Integer, j::Integer
)
    @boundscheck checkbounds(A, i, j)
    return @inbounds A.shmem[_single_index(A, Int32(i), Int32(j))]
end
@propagate_inbounds @inline function Base.setindex!(
    A::SingleAccessMatrix, v, i::Integer, j::Integer
)
    @boundscheck checkbounds(A, i, j)
    @inbounds A.shmem[_single_index(A, Int32(i), Int32(j))] = v
    return A
end

@inline _register_line_width(::Val{M}, ::Val{N}, ::RowOriented) where {M,N} = M
@inline _register_line_width(::Val{M}, ::Val{N}, ::ColOriented) where {M,N} = N

# base is the number of lanes before the group: CUDA.jl shuffles use base+j.
@propagate_inbounds @inline function _register_group_mask(::Val{D}, base::Int32) where {D}
    _validate_compute_shape(Val(D), Val(D), Val(D))
    @boundscheck begin
        (0i32 <= base <= 32i32 - Int32(D) && base % Int32(D) == 0i32) || throw(
            ArgumentError("register base must start a complete D_MAX-wide lane group")
        )
    end
    return (typemax(UInt32) >>> (32i32 - Int32(D))) << base
end

"""
    RegisterMatrix(mv, Val(M), Val(N), Val(D_MAX), orientation, base, mask, d)
    RegisterMatrix{T}(Val(M), Val(N), Val(D_MAX), orientation, base, d)

A lane's owned line of a logical M×N matrix. Row orientation needs M registers and
col orientation N registers. `base` is zero-based and `d` one-based within the group;
`mask` must name exactly that complete group. The allocating constructor initializes
all offered entries to zero, including entries in lanes without a logical line.

Runtime base/mask/d checks are bounds checks. An `@inbounds` caller must first prove
that base starts a complete aligned D_MAX-wide group, mask covers exactly that group,
and 1 <= d <= D_MAX. This permits kernels with proven geometry to omit exception
paths that can otherwise require local stack memory. Shape and line-width validation
remain unconditional compile-time checks.

There is deliberately no general indexing implementation: cross-lane access requires
the explicit collective accessors (`ours`, `ours_write!`, and `theirs`).
"""
struct RegisterMatrix{T,M,N,D,O<:Union{RowOriented,ColOriented},L} <: AbstractMatrix{T}
    mv::MVector{L,T}
    base::Int32
    mask::UInt32
    d::Int32

    @propagate_inbounds @inline function RegisterMatrix(
        mv::MVector{L,T},
        ::Val{M},
        ::Val{N},
        ::Val{D},
        o::O,
        base::Int32,
        mask::UInt32,
        d::Int32,
    ) where {T,M,N,D,O<:Union{RowOriented,ColOriented},L}
        _validate_compute_shape(Val(M), Val(N), Val(D))
        L == _register_line_width(Val(M), Val(N), o) ||
            throw(ArgumentError("register line width does not match shape and orientation"))
        @boundscheck begin
            mask == _register_group_mask(Val(D), base) ||
                throw(ArgumentError("register mask must cover exactly its lane group"))
            1i32 <= d <= Int32(D) || throw(ArgumentError("invalid within-group lane index"))
        end
        return new{T,M,N,D,O,L}(mv, base, mask, d)
    end
end

@propagate_inbounds @inline function RegisterMatrix{T}(
    ::Val{M}, ::Val{N}, ::Val{D}, o::Union{RowOriented,ColOriented}, base::Int32, d::Int32
) where {T,M,N,D}
    _validate_compute_shape(Val(M), Val(N), Val(D))
    # StaticArrays/ntuple require Int lengths, even when device dims are Val{Int32}.
    width = Int(_register_line_width(Val(M), Val(N), o))
    mv = MVector{width,T}(ntuple(_ -> zero(T), Val(width)))
    return RegisterMatrix(
        mv, Val(M), Val(N), Val(D), o, base, _register_group_mask(Val(D), base), d
    )
end

@inline Base.size(::RegisterMatrix{T,M,N}) where {T,M,N} = (Int(M), Int(N))
@inline Base.IndexStyle(::Type{<:RegisterMatrix}) = IndexCartesian()
@inline orientation(::Type{<:RegisterMatrix{T,M,N,D,O}}) where {T,M,N,D,O} = O()
@inline orientation(A::Union{SingleAccessMatrix,DualAccessMatrix,RegisterMatrix}) =
    orientation(typeof(A))

# This shared-memory transfer reads a different logical line in each lane. Use
# indexed loads: `theirs` promises group-uniform indices, even for shared sources.
@inline function single_to_register!(
    dest::RegisterMatrix{T,M,N,D,RowOriented}, source::SingleAccessMatrix{T,M,N,D}
) where {T,M,N,D}
    @unroll for k in 1i32:Int32(M)
        @inbounds dest.mv[k] = dest.d <= Int32(N) ? source[k, dest.d] : zero(T)
    end
    return dest
end
@inline function single_to_register!(
    dest::RegisterMatrix{T,M,N,D,ColOriented}, source::SingleAccessMatrix{T,M,N,D}
) where {T,M,N,D}
    @unroll for k in 1i32:Int32(N)
        @inbounds dest.mv[k] = dest.d <= Int32(M) ? source[dest.d, k] : zero(T)
    end
    return dest
end

"""
    register_to_single!(dest, source, d)

Materialize a logical source view into an oriented single-access tile. A source
with the matching effective orientation uses direct owned-line accesses, retaining
audited wrapper semantics; an opposite orientation uses uniform full-group
broadcasts. The caller invokes this only for complete active matrix groups.
"""
@inline function register_to_single!(
    dest::SingleAccessMatrix{T,M,N,D,RowOriented}, source, d::Int32
) where {T,M,N,D}
    return _register_to_single!(dest, source, d, orientation(source))
end
@inline function register_to_single!(
    dest::SingleAccessMatrix{T,M,N,D,ColOriented}, source, d::Int32
) where {T,M,N,D}
    return _register_to_single!(dest, source, d, orientation(source))
end

# Oriented logical wrappers retain the direct line copy whenever their effective
# orientation matches the staging tile. `ours` supplies their triangular and
# transpose semantics. The opposite-orientation methods below materialize through
# uniform broadcasts.
@inline function _register_to_single!(
    dest::SingleAccessMatrix{T,M,N,D,RowOriented}, source, d::Int32, ::RowOriented
) where {T,M,N,D}
    if d <= Int32(N)
        @unroll for k in 1i32:Int32(M)
            ours_write!(dest, k, d, ours(source, k, d, RowAccess()), RowAccess())
        end
    end
    return dest
end
@inline function _register_to_single!(
    dest::SingleAccessMatrix{T,M,N,D,ColOriented}, source, d::Int32, ::ColOriented
) where {T,M,N,D}
    if d <= Int32(M)
        @unroll for k in 1i32:Int32(N)
            ours_write!(dest, k, d, ours(source, k, d, ColAccess()), ColAccess())
        end
    end
    return dest
end

@inline function _register_to_single!(
    dest::SingleAccessMatrix{T,M,N,D,RowOriented}, source, d::Int32, ::Orientation
) where {T,M,N,D}
    @unroll for j in 1i32:Int32(N)
        @unroll for i in 1i32:Int32(M)
            value = theirs(source, i, j)
            if d == j
                ours_write!(dest, i, d, value, RowAccess())
            end
        end
    end
    return dest
end
@inline function _register_to_single!(
    dest::SingleAccessMatrix{T,M,N,D,ColOriented}, source, d::Int32, ::Orientation
) where {T,M,N,D}
    @unroll for i in 1i32:Int32(M)
        @unroll for j in 1i32:Int32(N)
            value = theirs(source, i, j)
            if d == i
                ours_write!(dest, j, d, value, ColAccess())
            end
        end
    end
    return dest
end

# Traits describe wrapper ownership only; logical wrapper accessors are separate.
@inline orientation(::Type{<:Adjoint{T,P}}) where {T,P} = flip(orientation(P))
@inline orientation(::Type{<:Transpose{T,P}}) where {T,P} = flip(orientation(P))
@inline orientation(
    ::Type{
        <:Union{
            LowerTriangular{T,P},
            UpperTriangular{T,P},
            UnitLowerTriangular{T,P},
            UnitUpperTriangular{T,P},
        },
    },
) where {T,P} = orientation(P)
@inline orientation(
    A::Union{
        Adjoint,
        Transpose,
        LowerTriangular,
        UpperTriangular,
        UnitLowerTriangular,
        UnitUpperTriangular,
    },
) = orientation(typeof(A))

#######################
#### SHARED MATRIX ####
#######################

struct SharedMatrix{T,D1,D2,pad_interval} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
end

"""
A matrix that is shared across batches and backed by shared memory.

Since there is only one matrix, both rows and columns can be accessed in parallel using the
usual single access padding.
"""

function SharedMatrix(shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}) where {T,D}
    pad_interval = div(32i32, D & -D) * D
    return SharedMatrix{T,D,D,pad_interval}(shmem)
end
function SharedMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D1}, ::Val{D2}
) where {T,D1,D2}
    pad_interval = div(32i32, D1 & -D1) * D1
    return SharedMatrix{T,D1,D2,pad_interval}(shmem)
end

@inline orientation(::Type{<:SharedMatrix}) = BothOriented()
@inline orientation(A::SharedMatrix) = orientation(typeof(A))

Base.@propagate_inbounds @inline function Base.getindex(
    A::SharedMatrix{T,D1,D2,pad_interval}, i::Int32, j::Int32
) where {T,D1,D2,pad_interval}
    raw_idx = (j - 1i32) * D1 + i
    padding = (raw_idx - 1i32) ÷ pad_interval
    return A.shmem[raw_idx + padding]
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::SharedMatrix{T,D1,D2,pad_interval}, v::T, i::Int32, j::Int32
) where {T,D1,D2,pad_interval}
    raw_idx = (j - 1i32) * D1 + i
    padding = (raw_idx - 1i32) ÷ pad_interval
    return A.shmem[raw_idx + padding] = v
end

# Support regular Int indexing (needed for Adjoint and other wrappers)
@propagate_inbounds @inline function Base.getindex(A::SharedMatrix, i::Int, j::Int)
    return getindex(A, Int32(i), Int32(j))
end

####################################
#### WRAPPER OPERATION MATRICES ####
####################################

"""
Abstract DualAccessMatrix wrapper type
"""
abstract type DualAccessMatrixWrapper{T,D} <: AbstractMatrix{T} end

@inline orientation(::Type{<:DualAccessMatrixWrapper}) = BothOriented()
@inline orientation(A::DualAccessMatrixWrapper) = orientation(typeof(A))

Base.parent(A::DualAccessMatrixWrapper{T,D}) where {T,D} = A.parent
Base.size(A::DualAccessMatrixWrapper{T,D}) where {T,D} = size(parent(A))

"""
Default getters and setters, no-ops
"""
@inline function wrapper_get(
    A::DualAccessMatrixWrapper{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    return v
end

@inline function wrapper_set(
    A::DualAccessMatrixWrapper{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    return v
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::DualAccessMatrixWrapper{T,D}, i::Int32, j::Int32
) where {T,D}
    return wrapper_get(A, parent(A)[i, j], i, j)
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrixWrapper{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    return parent(A)[i, j] = wrapper_set(A, v, i, j)
end

"""
Wrapper op for I - A
"""
struct IAddSubGetterMatrix{T,D,P<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::P
    a::T
    b::T
end

@inline function IAddSubGetterMatrix(A::DualAccessMatrix{T,D}, a::T, b::T) where {T,D}
    return IAddSubGetterMatrix{T,D,typeof(A)}(A, a, b)
end
@inline function IAddSubGetterMatrix(A::AbstractMatrix{T}, a::T, b::T) where {T}
    return IAddSubGetterMatrix{T,size(A, 1),typeof(A)}(A, a, b)
end
@inline Base.parent(A::IAddSubGetterMatrix) = A.parent
@inline Base.size(A::IAddSubGetterMatrix) = size(parent(A))
@inline Base.IndexStyle(::Type{<:IAddSubGetterMatrix}) = IndexCartesian()
@inline orientation(::Type{<:IAddSubGetterMatrix{T,D,P}}) where {T,D,P} = orientation(P)
@inline orientation(A::IAddSubGetterMatrix) = orientation(typeof(A))

@propagate_inbounds @inline function Base.getindex(
    A::IAddSubGetterMatrix, i::Integer, j::Integer
)
    return wrapper_get(A, parent(A)[i, j], Int32(i), Int32(j))
end

# Preserve the old dual-backed getter's raw write behavior only. New generalized
# getter wrappers are read-only through the explicit compute-accessor interface.
@propagate_inbounds @inline function Base.setindex!(
    A::IAddSubGetterMatrix{T,D,P}, v::T, i::Int32, j::Int32
) where {T,D,P<:DualAccessMatrix}
    return parent(A)[i, j] = v
end

Base.@propagate_inbounds @inline function wrapper_get(
    A::IAddSubGetterMatrix{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    return (i == j) * one(T) * A.a + A.b * v
end

"""Write-transform wrapper: store `(i == j)*a + b*v`, read the parent unchanged."""
struct IAddSubSetterMatrix{T,D,P<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::P
    a::T
    b::T
end

@inline function IAddSubSetterMatrix(A::DualAccessMatrix{T,D}, a::T, b::T) where {T,D}
    return IAddSubSetterMatrix{T,D,typeof(A)}(A, a, b)
end
@inline function IAddSubSetterMatrix(A::AbstractMatrix{T}, a::T, b::T) where {T}
    return IAddSubSetterMatrix{T,size(A, 1),typeof(A)}(A, a, b)
end
@inline Base.parent(A::IAddSubSetterMatrix) = A.parent
@inline Base.size(A::IAddSubSetterMatrix) = size(parent(A))
@inline Base.IndexStyle(::Type{<:IAddSubSetterMatrix}) = IndexCartesian()
@inline orientation(::Type{<:IAddSubSetterMatrix{T,D,P}}) where {T,D,P} = orientation(P)
@inline orientation(A::IAddSubSetterMatrix) = orientation(typeof(A))

@propagate_inbounds @inline function Base.getindex(
    A::IAddSubSetterMatrix, i::Integer, j::Integer
)
    return parent(A)[i, j]
end
@propagate_inbounds @inline function Base.setindex!(
    A::IAddSubSetterMatrix{T}, v::T, i::Integer, j::Integer
) where {T}
    return parent(A)[i, j] = wrapper_set(A, v, Int32(i), Int32(j))
end

Base.@propagate_inbounds @inline function wrapper_set(
    A::IAddSubSetterMatrix{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    return (i == j) * one(T) * A.a + A.b * v
end

"""
Load a single matrix from global memory into shared memory using a single warp. Global
memory is accessed linearly and then padding is introduced in shared memory to avoid bank
conflicts.

Uses of this function (either individually or multiple uses across warps) should be followed
by a sync_threads() call.
"""
@inline function shared_matrix_load!(shmem, global_arr, ::Val{D}) where {D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)

    pad_interval = div(32i32, D & -D) * D
    @inbounds begin
        offset = 0i32
        while offset < D * D
            raw_idx = offset + lid - 1i32
            if raw_idx < D * D
                padded_amount = raw_idx ÷ pad_interval

                src_idx = raw_idx + 1i32
                dest_idx = raw_idx + 1i32 + padded_amount

                shmem[dest_idx] = global_arr[src_idx]
            end

            offset += 32i32
        end
    end

    return nothing
end

@inline function shared_matrix_load!(shmem, global_arr, ::Val{D1}, ::Val{D2}) where {D1,D2}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)

    pad_interval = div(32i32, D1 & -D1) * D1
    @inbounds @unroll for offset in (0i32):(32i32):(D1 * D2 - 1)
        raw_idx = offset + lid
        if raw_idx <= D1 * D2
            padded_amount = (raw_idx - 1i32) ÷ pad_interval

            src_idx = raw_idx
            dest_idx = raw_idx + padded_amount

            shmem[dest_idx] = global_arr[src_idx]
        end
    end

    return nothing
end

@inline Base.size(::SharedMatrix{T,D1,D2,pad_interval}) where {T,D1,D2,pad_interval} =
    (D1, D2)
@inline Base.length(::SharedMatrix{T,D1,D2,pad_interval}) where {T,D1,D2,pad_interval} =
    D1 * D2
@inline Base.IndexStyle(::SharedMatrix) = IndexCartesian()

#######################
#### SHARED VECTOR ####
#######################

struct SharedVector{T,D} <: AbstractVector{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
end

"""A vector that is shared across batches and backed by shared memory."""
function SharedVector(shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}) where {T,D}
    return SharedVector{T,D}(shmem)
end

@propagate_inbounds @inline function Base.getindex(
    v::SharedVector{T,D}, i::Int32
) where {T,D}
    return v.shmem[i]
end
@propagate_inbounds @inline function Base.setindex!(
    v::SharedVector{T,D}, val::T, i::Int32
) where {T,D}
    return v.shmem[i] = val
end

@inline Base.size(::SharedVector{T,D}) where {T,D} = (D,)
@inline Base.length(::SharedVector{T,D}) where {T,D} = D
@inline Base.IndexStyle(::Type{<:SharedVector}) = IndexLinear()

"""
Load a single vector from global memory into shared memory using a single warp.

Uses of this function (either individually or multiple uses across warps) should be followed
by a sync_threads() call.
"""
@inline function shared_vector_load!(shmem, global_arr, ::Val{D}) where {D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)

    @inbounds if lid <= D
        shmem[lid] = global_arr[lid]
    end

    return nothing
end

#####################################
#### EXPLICIT MEMORY SUB-KERNELS ####
#####################################

"""
Function for loading matrices from global memory to shared memory, for the case
where one warp handles multiple matrices. Meant for small matrices.
"""
@inline function intermediate_layout_load!(
    shmem, global_arr, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32  # Works if nthreads is divisible by 32, otherwise needs to be cld(nthreads, 32)
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D * D  # Shared memory used my one warp (excluding padding for dual)
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32  # Global start idx for global memory
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32  # Global start for shared memory

    # Each thread loads warp_shmem_elems starting from start
    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
            raw_idx = start_raw + offset + lid - 1i32  # How many-th element to load in the block: [1, n_elements_per_block]
            raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32  # How many-th matrix to load: [1, n_mats_per_block]
            grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block  # How many-th global matrix 1 ... N to load

            if (
                raw_mtrx <= n_mats_per_block &&
                grid_mtrx_load <= N &&
                raw_idx <= warp_shmem_elem * wid
            )
                padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

                src_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
                dest_idx = start_offset + offset + lid - 1i32 + padded_amount

                shmem[dest_idx] = global_arr[src_idx]
            end

            offset += 32i32
        end
    end

    return nothing
end

"""
Only load lower triangular part, for Kalman filter
"""
@inline function intermediate_layout_load!(
    shmem, global_arr, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:lower}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32  # Works if nthreads is divisible by 32, otherwise needs to be cld(nthreads, 32)
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D * D  # Shared memory used my one warp (excluding padding for dual)
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32  # Global start idx for global memory
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32  # Global start for shared memory

    # Each thread loads warp_shmem_elems starting from start
    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
            raw_idx = start_raw + offset + lid - 1i32  # How many-th element to load in the block: [1, n_elements_per_block]
            raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32  # How many-th matrix to load: [1, n_mats_per_block]
            grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block  # How many-th global matrix 1 ... N to load

            i = mod1(offset + lid, D)
            j = (raw_idx - 1i32 - (raw_mtrx - 1i32) * D * D) ÷ D + 1i32

            if raw_mtrx <= n_mats_per_block &&
                grid_mtrx_load <= N &&
                raw_idx <= warp_shmem_elem * wid &&
                i >= j
                raw_idx_sym =
                    (mod1(raw_mtrx, n_mats_per_warp) - 1i32) * D * D + j + (i - 1i32) * D

                padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq
                padded_amount_sym = (raw_idx_sym - 1i32) ÷ interm_pad_freq

                src_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
                dest_idx = start_offset + offset + lid - 1i32 + padded_amount
                dest_idx_sym = start_offset + raw_idx_sym - 1i32 + padded_amount_sym

                val = global_arr[src_idx]

                shmem[dest_idx] = val
                shmem[dest_idx_sym] = val
            end

            offset += 32i32
        end
    end

    return nothing
end

"""
Function for writing matrices from shared memory to global memory, for the case
where one warp handles multiple matrices. Meant for small matrices.

This is the warp independent version, where one warp only writes matrices that
the warp was responsible for in previous calculations. This is the default mode
when no `Val(mode)` tag is supplied.
"""
@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    return intermediate_layout_write!(
        global_arr, shmem, Val(D), Val(nthreads), N, Val(:indep)
    )
end

@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:indep}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D * D
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32

    # Each thread loads warp_shmem_elems starting from start
    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
            raw_idx = start_raw + offset + lid - 1i32
            raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32
            grid_mtrx_write = raw_mtrx + (bid - 1i32) * n_mats_per_block

            if raw_mtrx <= n_mats_per_block &&
                grid_mtrx_write <= N &&
                raw_idx <= warp_shmem_elem * wid  # div(raw_idx - 1, warp_shmem_elem) != wid
                padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

                dest_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
                src_idx = start_offset + offset + lid - 1i32 + padded_amount

                global_arr[dest_idx] = shmem[src_idx]
            end

            offset += 32i32
        end
    end

    return nothing
end

"""
Triangular write, for Kalman filter
"""
@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:indep}, ::Val{:lower}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D * D
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32

    # Each thread loads warp_shmem_elems starting from start
    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
            raw_idx = start_raw + offset + lid - 1i32
            raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32
            grid_mtrx_write = raw_mtrx + (bid - 1i32) * n_mats_per_block

            i = mod1(offset + lid, D)
            j = (raw_idx - 1i32 - (raw_mtrx - 1i32) * D * D) ÷ D + 1i32

            if raw_mtrx <= n_mats_per_block &&
                grid_mtrx_write <= N &&
                raw_idx <= warp_shmem_elem * wid &&
                i >= j
                padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

                dest_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
                src_idx = start_offset + offset + lid - 1i32 + padded_amount

                global_arr[dest_idx] = shmem[src_idx]
            end

            offset += 32i32
        end
    end

    return nothing
end

"""
Function for writing matrices from shared memory to global memory, for the case
where one warp handles multiple matrices. Meant for small matrices.

This is the consequtive version, threads consequtively read from shared memory
and write to global memory, therefore also touching matrices that the specific
warp wasn't responsible in earlier calculations.
"""
@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:conseq}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    n_elements_per_block = n_mats_per_block * D * D

    tid = threadIdx().x
    bid = blockIdx().x

    sync_threads()

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D * D
    interm_pad_freq = div(32i32, D & -D) * D
    base_addr = (bid - 1i32) * n_mats_per_block * D * D + 1i32
    align_offset = (base_addr - 1i32) % 32i32

    @inbounds begin
        offset = 0i32
        while offset < n_elements_per_block + align_offset
            raw_idx = offset + tid - align_offset
            raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32
            grid_mtrx_store = raw_mtrx + (bid - 1i32) * n_mats_per_block

            if raw_mtrx <= n_mats_per_block && grid_mtrx_store <= N && raw_idx > 0i32
                dest_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx

                elem_warp_id = (raw_idx - 1i32) % warp_shmem_elem + 1
                padded_amount = (elem_warp_id - 1i32) ÷ interm_pad_freq
                warp_offset = ((raw_idx - 1i32) ÷ warp_shmem_elem) * warp_shmem_size
                src_idx = warp_offset + elem_warp_id + padded_amount

                global_arr[dest_idx] = shmem[src_idx]
            end

            offset += nthreads
        end
    end

    sync_threads()

    return nothing
end

"""
Function that transfers matrices from intermediate layout to dual memory layout.
This is for the case where one warp handles multiple matrices, meant for small matrices.
"""
@inline function interm_to_dual_transfer!(
    shmem_dual, shmem_interm, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + dual_padding
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    interm_pad_freq = div(32i32, D & -D) * D

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    matrix_thread = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    @inbounds if lid <= active_lanes && grid_matrix_id <= N
        col = matrix_thread

        # Loop over the rows of each matrix
        for row in (1i32):D
            # Compute index for intermediate layout
            logical_idx = (warp_matrix_id - 1i32) * D * D + (col - 1i32) * D + row
            padding = (logical_idx - 1i32) ÷ interm_pad_freq
            padded_idx_interm = logical_idx + padding + (wid - 1i32) * warp_shmem_size

            # Compute index for dual-access layout
            padded_idx_dual = (
                (col - 1i32 + (wid - 1i32) * D) * stride +
                (row - 1i32) * n_mats_per_warp +
                warp_matrix_id - (wid - 1i32) * dual_padding
            )

            # Load from intermediate layout to dual-access layout
            shmem_dual[padded_idx_dual] = shmem_interm[padded_idx_interm]
        end
    end

    return nothing
end

"""
Function that transfers matrices from dual layout to intermediate memory layout.
This is for the case where one warp handles multiple matrices, meant for small matrices.
"""
@inline function dual_to_interm_transfer!(
    shmem_interm, shmem_dual, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + dual_padding
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    interm_pad_freq = div(32i32, D & -D) * D

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    matrix_thread = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    @inbounds if lid <= active_lanes && grid_matrix_id <= N
        col = matrix_thread

        # Loop over the rows of each matrix
        for row in (1i32):D
            # Compute index for intermediate layout
            logical_idx = (warp_matrix_id - 1i32) * D * D + (col - 1i32) * D + row
            padding = (logical_idx - 1i32) ÷ interm_pad_freq
            padded_idx_interm = logical_idx + padding + (wid - 1i32) * warp_shmem_size

            # Compute index for dual-access layout
            padded_idx_dual = (
                (col - 1i32 + (wid - 1i32) * D) * stride +
                (row - 1i32) * n_mats_per_warp +
                warp_matrix_id - (wid - 1i32) * dual_padding
            )

            # Load from intermediate layout to dual-access layout
            shmem_interm[padded_idx_interm] = shmem_dual[padded_idx_dual]
        end
    end

    return nothing
end

@inline function intermediate_layout_load!(
    shmem, global_arr, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,D1,D2,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D1 * D2
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32

    # offset = 0i32
    # while offset < warp_shmem_elem
    @inbounds @unroll for o in (1i32):cld(warp_shmem_elem, 32i32)
        offset = (o - 1i32) * 32i32
        raw_idx = start_raw + offset + lid - 1i32
        raw_mtrx = div(raw_idx - 1i32, D1 * D2) + 1i32
        grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block

        if (
            raw_mtrx <= n_mats_per_block &&
            grid_mtrx_load <= N &&
            raw_idx <= warp_shmem_elem * wid
        )
            padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

            src_idx = (bid - 1i32) * n_mats_per_block * D1 * D2 + raw_idx
            dest_idx = start_offset + offset + lid - 1i32 + padded_amount

            shmem[dest_idx] = global_arr[src_idx]
        end

        # offset += 32i32
    end

    return nothing
end

@inline function intermediate_layout_load!(
    shmem,
    global_arr,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    ::Val{n_mats_per_block},
    N::Int32,
) where {D,D1,D2,n_mats_per_block,nthreads}
    n_mats_per_warp = 32i32 ÷ D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D1 * D2
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32

    # offset = 0i32
    # while offset < warp_shmem_elem
    @inbounds @unroll for o in (1i32):cld(warp_shmem_elem, 32i32)
        offset = (o - 1i32) * 32i32
        raw_idx = start_raw + offset + lid - 1i32
        raw_mtrx = div(raw_idx - 1i32, D1 * D2) + 1i32
        grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block

        if (
            raw_mtrx <= n_mats_per_block &&
            grid_mtrx_load <= N &&
            raw_idx <= warp_shmem_elem * wid
        )
            padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

            src_idx = (bid - 1i32) * n_mats_per_block * D1 * D2 + raw_idx
            dest_idx = start_offset + offset + lid - 1i32 + padded_amount

            shmem[dest_idx] = global_arr[src_idx]
        end

        # offset += 32i32
    end

    return nothing
end

@inline function interm_to_dual_transfer!(
    shmem_dual, shmem_interm, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,D1,D2,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + dual_padding
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    interm_pad_freq = div(32i32, D & -D) * D

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    col = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    if lid <= active_lanes && grid_matrix_id <= N && col <= D2
        @inbounds @unroll for row in (1i32):D1
            logical_idx = (warp_matrix_id - 1i32) * D1 * D2 + (col - 1i32) * D1 + row
            padding = (logical_idx - 1i32) ÷ interm_pad_freq
            padded_idx_interm = logical_idx + padding + (wid - 1i32) * warp_shmem_size

            padded_idx_dual = (
                (col - 1i32 + (wid - 1i32) * D) * stride +
                (row - 1i32) * n_mats_per_warp +
                warp_matrix_id - (wid - 1i32) * dual_padding
            )

            shmem_dual[padded_idx_dual] = shmem_interm[padded_idx_interm]
        end
    end

    return nothing
end

@inline function interm_to_dual_transfer!(
    shmem_dual,
    shmem_interm,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    ::Val{n_mats_per_block},
    N::Int32,
) where {D,D1,D2,n_mats_per_block,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + dual_padding
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    interm_pad_freq = div(32i32, D & -D) * D

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    col = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    if lid <= active_lanes && grid_matrix_id <= N && col <= D2
        @inbounds @unroll for row in (1i32):D1
            logical_idx = (warp_matrix_id - 1i32) * D1 * D2 + (col - 1i32) * D1 + row
            padding = (logical_idx - 1i32) ÷ interm_pad_freq
            padded_idx_interm = logical_idx + padding + (wid - 1i32) * warp_shmem_size

            padded_idx_dual = (
                (col - 1i32 + (wid - 1i32) * D) * stride +
                (row - 1i32) * n_mats_per_warp +
                warp_matrix_id - (wid - 1i32) * dual_padding
            )

            shmem_dual[padded_idx_dual] = shmem_interm[padded_idx_interm]
        end
    end

    return nothing
end

"""
Rectangular `dual_to_interm_transfer!` (`(D1,D2)` over dual-layout slot dim
`D`). The second argument `M_dual` is accessed as `M_dual[row, col]`, i.e. as
any 2D-indexable view of the dual slot: `DualAccessMatrix` for a dense write,
or a stdlib wrapper like `UpperTriangular(view)`, `LowerTriangular(view)`,
`Adjoint(view)` to write only that structural part (the wrapper's `getindex`
returns zero for masked positions). This asymmetry vs the load path —
`interm_to_dual_transfer!` reads raw shmem — is deliberate: structured *output*
views appear in QR, triangular Kalman writes, etc., whereas the load path
always pulls plain dense blocks from global memory.
"""
@inline function dual_to_interm_transfer!(
    shmem_interm, M_dual, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,D1,D2,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    interm_pad_freq = div(32i32, D & -D) * D

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    col = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    if warp_matrix_id <= n_mats_per_warp && grid_matrix_id <= N && col <= D2
        @inbounds @unroll for row in (1i32):D1
            logical_idx = (warp_matrix_id - 1i32) * D1 * D2 + (col - 1i32) * D1 + row
            padding = (logical_idx - 1i32) ÷ interm_pad_freq
            padded_idx_interm = logical_idx + padding + (wid - 1i32) * warp_shmem_size

            shmem_interm[padded_idx_interm] = M_dual[row, col]
        end
    end

    return nothing
end

"""
As above, but with a runtime `Val(n_mats_per_block)` overriding the value
derived from `nthreads ÷ 32 * (32 ÷ D)`. Same wrapper-accepting `M_dual`
contract as the no-`n_mats_per_block` variant.
"""
@inline function dual_to_interm_transfer!(
    shmem_interm,
    M_dual,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    ::Val{n_mats_per_block},
    N::Int32,
) where {D,D1,D2,n_mats_per_block,nthreads}
    n_mats_per_warp = 32i32 ÷ D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    dual_padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    interm_pad_freq = div(32i32, D & -D) * D

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    col = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    if warp_matrix_id <= n_mats_per_warp && grid_matrix_id <= N && col <= D2
        @inbounds @unroll for row in (1i32):D1
            logical_idx = (warp_matrix_id - 1i32) * D1 * D2 + (col - 1i32) * D1 + row
            padding = (logical_idx - 1i32) ÷ interm_pad_freq
            padded_idx_interm = logical_idx + padding + (wid - 1i32) * warp_shmem_size

            shmem_interm[padded_idx_interm] = M_dual[row, col]
        end
    end

    return nothing
end

@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,D1,D2,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D1 * D2
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32

    # offset = 0i32
    # while offset < warp_shmem_elem
    @inbounds @unroll for o in (1i32):cld(warp_shmem_elem, 32i32)
        offset = (o - 1i32) * 32i32
        raw_idx = start_raw + offset + lid - 1i32
        raw_mtrx = div(raw_idx - 1i32, D1 * D2) + 1i32
        grid_mtrx_write = raw_mtrx + (bid - 1i32) * n_mats_per_block

        if (
            raw_mtrx <= n_mats_per_block &&
            grid_mtrx_write <= N &&
            raw_idx <= warp_shmem_elem * wid
        )
            padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

            dest_idx = (bid - 1i32) * n_mats_per_block * D1 * D2 + raw_idx
            src_idx = start_offset + offset + lid - 1i32 + padded_amount

            global_arr[dest_idx] = shmem[src_idx]
        end

        # offset += 32i32
    end

    return nothing
end

@inline function intermediate_layout_write!(
    global_arr,
    shmem,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    ::Val{n_mats_per_block},
    N::Int32,
) where {D,D1,D2,n_mats_per_block,nthreads}
    n_mats_per_warp = 32i32 ÷ D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(32i32 ÷ D - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = n_mats_per_warp * D * D + padding * (D - 1i32)
    warp_shmem_elem = n_mats_per_warp * D1 * D2
    interm_pad_freq = div(32i32, D & -D) * D
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32

    # offset = 0i32
    # while offset < warp_shmem_elem
    @inbounds @unroll for o in (1i32):cld(warp_shmem_elem, 32i32)
        offset = (o - 1i32) * 32i32
        raw_idx = start_raw + offset + lid - 1i32
        raw_mtrx = div(raw_idx - 1i32, D1 * D2) + 1i32
        grid_mtrx_write = raw_mtrx + (bid - 1i32) * n_mats_per_block

        if (
            raw_mtrx <= n_mats_per_block &&
            grid_mtrx_write <= N &&
            raw_idx <= warp_shmem_elem * wid
        )
            padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

            dest_idx = (bid - 1i32) * n_mats_per_block * D1 * D2 + raw_idx
            src_idx = start_offset + offset + lid - 1i32 + padded_amount

            global_arr[dest_idx] = shmem[src_idx]
        end

        # offset += 32i32
    end

    return nothing
end

@inline function scalar_stage!(
    shmem,
    val::T,
    lid::Int32,
    warp_matrix_id::Int32,
    block_matrix_id::Int32,
    active::Bool,
    ::Val{D},
) where {T,D}
    is_leader = lid == (warp_matrix_id - 1i32) * D + 1i32
    if is_leader && active
        shmem[block_matrix_id] = val
    end
end

@inline function scalar_write!(global_arr, shmem, n_mats_per_block::Int32, N::Int32)
    tid = threadIdx().x
    base = (blockIdx().x - 1i32) * n_mats_per_block
    if tid <= n_mats_per_block && (base + tid) <= N
        @inbounds global_arr[base + tid] = shmem[tid]
    end
end

##########################################
#### PADDED SINGLE COMPUTE TRANSFERS ######
##########################################

"""
    intermediate_layout_load!(single, global, Val(M), Val(N), Val(D), Val(nthreads), count, orientation)

Cooperatively load column-major logical M×N matrices into new padded D×D single
compute tiles. Each warp pass reads consecutive valid global elements, then scatters
by logical coordinates. Both orientations preserve global coalescing; shared scatter
bank costs depend on shape and orientation. Only valid logical entries are written.

All new orientation-taking transfers require the launch block size to equal
`nthreads`, with complete warps. They contain no barriers: callers must synchronize
the full warp after cooperative production and before readers, and before reusing
storage still read by another lane. Single/dual conversion buffers must not overlap.
Call cooperative global transfers outside per-matrix active guards; internal guards
handle batch tails and unused lanes.
"""
@inline function intermediate_layout_load!(
    shmem,
    global_arr,
    ::Val{M},
    ::Val{N},
    ::Val{D},
    ::Val{nthreads},
    count::Int32,
    o::Union{RowOriented,ColOriented},
) where {M,N,D,nthreads}
    _validate_compute_shape(Val(M), Val(N), Val(D))
    _validate_layout_threads(Val(nthreads))
    tid = threadIdx().x
    wid = (tid - 1i32) ÷ 32i32 + 1i32
    # Masking exposes the nonnegative lane range to transfer-index simplification.
    lid = ((tid - 1i32) & 31i32) + 1i32
    nmat = 32i32 ÷ Int32(D)
    block_mats = (Int32(nthreads) ÷ 32i32) * nmat
    first_mat = (blockIdx().x - 1i32) * block_mats + (wid - 1i32) * nmat
    matrix_elems = Int32(M) * Int32(N)
    warp_elems = nmat * matrix_elems
    @inbounds @unroll for pass in (1i32):cld(warp_elems, 32i32)
        r = (pass - 1i32) * 32i32 + lid - 1i32
        mid = r ÷ matrix_elems + 1i32
        if r < warp_elems && first_mat + mid <= count
            elem = r % matrix_elems
            i = elem % Int32(M) + 1i32
            j = elem ÷ Int32(M) + 1i32
            dest = _single_transfer_address(Val(D), o, wid, mid, i, j)
            shmem[dest] = global_arr[first_mat * matrix_elems + r + 1i32]
        end
    end
    return nothing
end

"""Inverse of the orientation-taking cooperative load; the same synchronization contract applies."""
@inline function intermediate_layout_write!(
    global_arr,
    shmem,
    ::Val{M},
    ::Val{N},
    ::Val{D},
    ::Val{nthreads},
    count::Int32,
    o::Union{RowOriented,ColOriented},
) where {M,N,D,nthreads}
    _validate_compute_shape(Val(M), Val(N), Val(D))
    _validate_layout_threads(Val(nthreads))
    tid = threadIdx().x
    wid = (tid - 1i32) ÷ 32i32 + 1i32
    # Masking exposes the nonnegative lane range to transfer-index simplification.
    lid = ((tid - 1i32) & 31i32) + 1i32
    nmat = 32i32 ÷ Int32(D)
    block_mats = (Int32(nthreads) ÷ 32i32) * nmat
    first_mat = (blockIdx().x - 1i32) * block_mats + (wid - 1i32) * nmat
    matrix_elems = Int32(M) * Int32(N)
    warp_elems = nmat * matrix_elems
    @inbounds @unroll for pass in (1i32):cld(warp_elems, 32i32)
        r = (pass - 1i32) * 32i32 + lid - 1i32
        mid = r ÷ matrix_elems + 1i32
        if r < warp_elems && first_mat + mid <= count
            elem = r % matrix_elems
            i = elem % Int32(M) + 1i32
            j = elem ÷ Int32(M) + 1i32
            src = _single_transfer_address(Val(D), o, wid, mid, i, j)
            global_arr[first_mat * matrix_elems + r + 1i32] = shmem[src]
        end
    end
    return nothing
end

@inline _owned_coords(k::Int32, d::Int32, ::RowOriented) = (k, d)
@inline _owned_coords(k::Int32, d::Int32, ::ColOriented) = (d, k)
@inline _owned_line_count(::Val{M}, ::Val{N}, ::RowOriented) where {M,N} = Int32(N)
@inline _owned_line_count(::Val{M}, ::Val{N}, ::ColOriented) where {M,N} = Int32(M)

"""
Copy new single compute storage into disjoint raw dual storage. Each lane copies its
owned line according to the single orientation. Only valid logical entries are
written; caller warp synchronization is required before consumers or storage reuse.
"""
@inline function interm_to_dual_transfer!(
    shmem_dual,
    shmem_single,
    ::Val{M},
    ::Val{N},
    ::Val{D},
    ::Val{nthreads},
    count::Int32,
    o::Union{RowOriented,ColOriented},
) where {M,N,D,nthreads}
    _validate_compute_shape(Val(M), Val(N), Val(D))
    _validate_layout_threads(Val(nthreads))
    tid = threadIdx().x
    wid = (tid - 1i32) ÷ 32i32 + 1i32
    lid = mod1(tid, 32i32)
    nmat = 32i32 ÷ Int32(D)
    mid = (lid - 1i32) ÷ Int32(D) + 1i32
    d = mod1(lid, Int32(D))
    block_mats = (Int32(nthreads) ÷ 32i32) * nmat
    global_mat = (blockIdx().x - 1i32) * block_mats + (wid - 1i32) * nmat + mid
    if mid <= nmat && global_mat <= count && d <= _owned_line_count(Val(M), Val(N), o)
        dual_outer = (wid - 1i32) * _dual_warp_stride(Val(D))
        @inbounds @unroll for k in (1i32):Int32(_register_line_width(Val(M), Val(N), o))
            i, j = _owned_coords(k, d, o)
            src = _single_transfer_address(Val(D), o, wid, mid, i, j)
            dest =
                dual_outer +
                mid +
                (j - 1i32) * Int32(_compute_stride(Val(D))) +
                (i - 1i32) * nmat
            shmem_dual[dest] = shmem_single[src]
        end
    end
    return nothing
end

"""
Copy a logical dual matrix view into disjoint new single compute storage. `M_dual`
must denote the calling lane group's matrix and support logical two-index reads;
wrappers are read through their indexing semantics to materialize structural zeros
or transposes. Logical M,N refer to the wrapped view's output shape. Caller warp
synchronization is required before this copy and before cooperative output reads.
"""
@inline function dual_to_interm_transfer!(
    shmem_single,
    M_dual,
    ::Val{M},
    ::Val{N},
    ::Val{D},
    ::Val{nthreads},
    count::Int32,
    o::Union{RowOriented,ColOriented},
) where {M,N,D,nthreads}
    _validate_compute_shape(Val(M), Val(N), Val(D))
    _validate_layout_threads(Val(nthreads))
    tid = threadIdx().x
    wid = (tid - 1i32) ÷ 32i32 + 1i32
    lid = mod1(tid, 32i32)
    nmat = 32i32 ÷ Int32(D)
    mid = (lid - 1i32) ÷ Int32(D) + 1i32
    d = mod1(lid, Int32(D))
    block_mats = (Int32(nthreads) ÷ 32i32) * nmat
    global_mat = (blockIdx().x - 1i32) * block_mats + (wid - 1i32) * nmat + mid
    if mid <= nmat && global_mat <= count && d <= _owned_line_count(Val(M), Val(N), o)
        @inbounds @unroll for k in (1i32):Int32(_register_line_width(Val(M), Val(N), o))
            i, j = _owned_coords(k, d, o)
            dest = _single_transfer_address(Val(D), o, wid, mid, i, j)
            shmem_single[dest] = M_dual[i, j]
        end
    end
    return nothing
end
