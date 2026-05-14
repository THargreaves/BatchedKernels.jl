import Base: @propagate_inbounds
import LinearAlgebra: AdjOrTransAbsMat, wrapperop

export ConflictDualAccessMatrix, ConflictSharedMatrix, ConflictIAddSubSetterMatrix, ConflictIAddSubGetterMatrix
export ConflictSharedVector, ConflictBatchedVector
export conflict_batched_layout_load!, conflict_batched_layout_write!
export conflict_shared_matrix_load!, conflict_shared_vector_load!
export conflict_vector_load!, conflict_vector_write!

#############################
#### BATCHED MATRIX TYPE ####
#############################

struct ConflictDualAccessMatrix{T,D} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
    offset::Int32
end

function ConflictDualAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, warp_matrix_id::Int32, ::Val{:small},
) where {T,D}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    n_mats_per_warp = 32i32 ÷ D
    offset = ((wid - 1i32) * n_mats_per_warp + (warp_matrix_id - 1i32)) * D * D

    return ConflictDualAccessMatrix{T,D}(shmem, offset)
end

function ConflictDualAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, wid::Int32, warp_matrix_id::Int32, ::Val{:small},
) where {T,D}
    n_mats_per_warp = 32i32 ÷ D
    offset = ((wid - 1i32) * n_mats_per_warp + (warp_matrix_id - 1i32)) * D * D
    return ConflictDualAccessMatrix{T,D}(shmem, offset)
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::ConflictDualAccessMatrix{T,D}, i::Int32, j::Int32,
) where {T,D}
    return A.shmem[A.offset + (j - 1i32) * D + i]
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::ConflictDualAccessMatrix{T,D}, v::T, i::Int32, j::Int32,
) where {T,D}
    return A.shmem[A.offset + (j - 1i32) * D + i] = v
end

@propagate_inbounds @inline function Base.getindex(A::ConflictDualAccessMatrix, i::Int, j::Int)
    return getindex(A, Int32(i), Int32(j))
end

@propagate_inbounds @inline function Base.setindex!(A::ConflictDualAccessMatrix{T}, v::T, i::Int, j::Int) where {T}
    return setindex!(A, v, Int32(i), Int32(j))
end

@inline Base.size(::ConflictDualAccessMatrix{T,D}) where {T,D} = (D, D)
@inline Base.length(::ConflictDualAccessMatrix{T,D}) where {T,D} = D * D
@inline Base.IndexStyle(::Type{<:ConflictDualAccessMatrix}) = IndexCartesian()

#######################
#### SHARED MATRIX ####
#######################

struct ConflictSharedMatrix{T,D1,D2} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
end

function ConflictSharedMatrix(shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}) where {T,D}
    return ConflictSharedMatrix{T,D,D}(shmem)
end

function ConflictSharedMatrix(shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D1}, ::Val{D2}) where {T,D1,D2}
    return ConflictSharedMatrix{T,D1,D2}(shmem)
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::ConflictSharedMatrix{T,D1,D2}, i::Int32, j::Int32,
) where {T,D1,D2}
    return A.shmem[(j - 1i32) * D1 + i]
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::ConflictSharedMatrix{T,D1,D2}, v::T, i::Int32, j::Int32,
) where {T,D1,D2}
    return A.shmem[(j - 1i32) * D1 + i] = v
end

@propagate_inbounds @inline function Base.getindex(A::ConflictSharedMatrix, i::Int, j::Int)
    return getindex(A, Int32(i), Int32(j))
end

@inline Base.size(::ConflictSharedMatrix{T,D1,D2}) where {T,D1,D2} = (D1, D2)
@inline Base.length(::ConflictSharedMatrix{T,D1,D2}) where {T,D1,D2} = D1 * D2
@inline Base.IndexStyle(::ConflictSharedMatrix) = IndexCartesian()

####################################
#### WRAPPER OPERATION MATRICES ####
####################################

abstract type ConflictDualAccessMatrixWrapper{T,D} <: AbstractMatrix{T} end

Base.parent(A::ConflictDualAccessMatrixWrapper) = A.parent
@inline Base.size(::ConflictDualAccessMatrixWrapper{T,D}) where {T,D} = (D, D)

@inline function wrapper_get(::ConflictDualAccessMatrixWrapper{T,D}, v::T, ::Int32, ::Int32) where {T,D}
    return v
end

@inline function wrapper_set(::ConflictDualAccessMatrixWrapper{T,D}, v::T, ::Int32, ::Int32) where {T,D}
    return v
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::ConflictDualAccessMatrixWrapper{T,D}, i::Int32, j::Int32,
) where {T,D}
    return wrapper_get(A, parent(A)[i, j], i, j)
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::ConflictDualAccessMatrixWrapper{T,D}, v::T, i::Int32, j::Int32,
) where {T,D}
    return parent(A)[i, j] = wrapper_set(A, v, i, j)
end

struct ConflictIAddSubGetterMatrix{T,D} <: ConflictDualAccessMatrixWrapper{T,D}
    parent::ConflictDualAccessMatrix{T,D}
    a::T
    b::T
end

Base.@propagate_inbounds @inline function wrapper_get(
    A::ConflictIAddSubGetterMatrix{T,D}, v::T, i::Int32, j::Int32,
) where {T,D}
    return (i == j) * one(T) * A.a + A.b * v
end

struct ConflictIAddSubSetterMatrix{T,D} <: ConflictDualAccessMatrixWrapper{T,D}
    parent::ConflictDualAccessMatrix{T,D}
    a::T
    b::T
end

Base.@propagate_inbounds @inline function wrapper_set(
    A::ConflictIAddSubSetterMatrix{T,D}, v::T, i::Int32, j::Int32,
) where {T,D}
    return (i == j) * one(T) * A.a + A.b * v
end

########################
#### BATCHED VECTOR ####
########################

struct ConflictBatchedVector{T,D} <: AbstractVector{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
    offset::Int32
end

function ConflictBatchedVector(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, warp_vector_id::Int32,
) where {T,D}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    n_vecs_per_warp = 32i32 ÷ D
    offset = ((wid - 1i32) * n_vecs_per_warp + (warp_vector_id - 1i32)) * D
    return ConflictBatchedVector{T,D}(shmem, offset)
end

Base.@propagate_inbounds @inline function Base.getindex(v::ConflictBatchedVector{T,D}, i::Int32) where {T,D}
    return v.shmem[v.offset + i]
end

Base.@propagate_inbounds @inline function Base.setindex!(v::ConflictBatchedVector{T,D}, val::T, i::Int32) where {T,D}
    return v.shmem[v.offset + i] = val
end

@inline Base.size(::ConflictBatchedVector{T,D}) where {T,D} = (D,)
@inline Base.length(::ConflictBatchedVector{T,D}) where {T,D} = D
@inline Base.IndexStyle(::Type{<:ConflictBatchedVector}) = IndexLinear()

#######################
#### SHARED VECTOR ####
#######################

struct ConflictSharedVector{T,D} <: AbstractVector{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
end

function ConflictSharedVector(shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}) where {T,D}
    return ConflictSharedVector{T,D}(shmem)
end

@propagate_inbounds @inline function Base.getindex(v::ConflictSharedVector{T,D}, i::Int32) where {T,D}
    return v.shmem[i]
end

@propagate_inbounds @inline function Base.setindex!(v::ConflictSharedVector{T,D}, val::T, i::Int32) where {T,D}
    return v.shmem[i] = val
end

@inline Base.size(::ConflictSharedVector{T,D}) where {T,D} = (D,)
@inline Base.length(::ConflictSharedVector{T,D}) where {T,D} = D
@inline Base.IndexStyle(::Type{<:ConflictSharedVector}) = IndexLinear()

######################################
#### SHARED MATRIX/VECTOR LOAD    ####
######################################

@inline function conflict_shared_matrix_load!(shmem, global_arr, ::Val{D1}, ::Val{D2}) where {D1,D2}
    lid = mod1(threadIdx().x, 32i32)
    @inbounds begin
        offset = 0i32
        while offset < D1 * D2
            idx = offset + lid
            if idx <= D1 * D2
                shmem[idx] = global_arr[idx]
            end
            offset += 32i32
        end
    end
    return nothing
end

@inline function conflict_shared_vector_load!(shmem, global_arr, ::Val{D}) where {D}
    lid = mod1(threadIdx().x, 32i32)
    @inbounds if lid <= D
        shmem[lid] = global_arr[lid]
    end
    return nothing
end

##############################
#### BATCHED MATRIX LOAD  ####
##############################

@inline function conflict_batched_layout_load!(
    shmem, global_arr, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small},
) where {D,D1,D2,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    warp_shmem_size = n_mats_per_warp * D * D
    shmem_warp_offset = (wid - 1i32) * warp_shmem_size

    warp_global_elems = n_mats_per_warp * D1 * D2
    global_warp_offset = ((bid - 1i32) * n_mats_per_block + (wid - 1i32) * n_mats_per_warp) * D1 * D2

    @inbounds begin
        offset = 0i32
        while offset < warp_global_elems
            idx = offset + lid
            if idx <= warp_global_elems
                mat_in_warp = div(idx - 1i32, D1 * D2)
                elem_in_mat = idx - mat_in_warp * D1 * D2
                col = div(elem_in_mat - 1i32, D1)
                row = elem_in_mat - col * D1

                global_mat = (bid - 1i32) * n_mats_per_block + (wid - 1i32) * n_mats_per_warp + mat_in_warp + 1i32

                if global_mat <= N
                    src = global_warp_offset + idx
                    dst = shmem_warp_offset + mat_in_warp * D * D + col * D + row
                    shmem[dst] = global_arr[src]
                end
            end
            offset += 32i32
        end
    end

    return nothing
end

###############################
#### BATCHED MATRIX WRITE  ####
###############################

@inline function conflict_batched_layout_write!(
    global_arr, shmem, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small},
) where {D,D1,D2,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    warp_shmem_size = n_mats_per_warp * D * D
    shmem_warp_offset = (wid - 1i32) * warp_shmem_size

    warp_global_elems = n_mats_per_warp * D1 * D2
    global_warp_offset = ((bid - 1i32) * n_mats_per_block + (wid - 1i32) * n_mats_per_warp) * D1 * D2

    @inbounds begin
        offset = 0i32
        while offset < warp_global_elems
            idx = offset + lid
            if idx <= warp_global_elems
                mat_in_warp = div(idx - 1i32, D1 * D2)
                elem_in_mat = idx - mat_in_warp * D1 * D2
                col = div(elem_in_mat - 1i32, D1)
                row = elem_in_mat - col * D1

                global_mat = (bid - 1i32) * n_mats_per_block + (wid - 1i32) * n_mats_per_warp + mat_in_warp + 1i32

                if global_mat <= N
                    src = shmem_warp_offset + mat_in_warp * D * D + col * D + row
                    dst = global_warp_offset + idx
                    global_arr[dst] = shmem[src]
                end
            end
            offset += 32i32
        end
    end

    return nothing
end

############################
#### VECTOR LOAD/WRITE  ####
############################

@inline function conflict_vector_load!(
    shmem, global_arr, ::Val{D1}, ::Val{D}, ::Val{nthreads}, N::Int32,
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

@inline function conflict_vector_write!(
    global_arr, shmem, ::Val{D1}, ::Val{D}, ::Val{nthreads}, N::Int32,
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
