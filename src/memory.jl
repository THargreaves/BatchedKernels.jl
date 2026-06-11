import Base: @propagate_inbounds
import LinearAlgebra: AdjOrTransAbsMat, wrapperop
using KernelAbstractions.Extras: @unroll

export DualAccessMatrix,
    SingleAccessMatrix,
    SharedMatrix,
    IAddSubSetterMatrix,
    IAddSubGetterMatrix,
    SharedVector,
    BatchedVector
export BlockMatrix_2_1, BlockMatrixLowerTrig_2_2
export intermediate_layout_load!, intermediate_layout_write!
export interm_to_dual_transfer!, dual_to_interm_transfer!
export shared_matrix_load!, shared_vector_load!
export vector_load!, vector_write!
export scalar_stage!, scalar_write!

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

struct SingleAccessMatrix{T,pad_interval} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
    outer_offset::Int32
    inner_offset::Int32
end

# TODO: shouldn't be baking in the warp_matrix_id if we want to use it with global memory
"""
A single access matrix view into shared memory.

Rows of the matrices can be accessed in parallel without bank conflicts. Padding is included
between warps to be compatible with the dual access layout without the need for
synchronisation. 
"""
function SingleAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, wid::Int32, warp_matrix_id::Int32
) where {T,D}
    n_mats_per_warp = 32i32 ÷ D

    dual_access_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    dual_access_stride = (n_mats_per_warp * D + dual_access_padding) * D

    outer_offset = dual_access_stride * (wid - 1i32)
    inner_offset = (warp_matrix_id - 1i32) * D^2

    pad_interval = div(32i32, D & -D) * D

    return SingleAccessMatrix{T,pad_interval}(shmem, outer_offset, inner_offset)
end

# TODO: replace div with magic number
@propagate_inbounds @inline function Base.getindex(
    A::SingleAccessMatrix{T,pad_interval}, i::Int32
) where {T,pad_interval}
    warp_idx = A.inner_offset + i
    padding = (warp_idx - 1i32) ÷ pad_interval
    return A.shmem[A.outer_offset + warp_idx + padding] = v
end

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
struct IAddSubGetterMatrix{T,D} <: DualAccessMatrixWrapper{T,D}
    parent::DualAccessMatrix{T,D}
    a::T
    b::T
end

Base.@propagate_inbounds @inline function wrapper_get(
    A::IAddSubGetterMatrix{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    return (i == j) * one(T) * A.a + A.b * v
end

struct IAddSubSetterMatrix{T,D} <: DualAccessMatrixWrapper{T,D}
    parent::DualAccessMatrix{T,D}
    a::T
    b::T
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
