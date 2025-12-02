import Base: @propagate_inbounds
import LinearAlgebra: AdjOrTransAbsMat, wrapperop

export DualAccessMatrix, SingleAccessMatrix, SharedMatrix, IMinusSetterMatrix, SharedVector, BatchedVector
export intermediate_layout_load!, intermediate_layout_write!
export interm_to_dual_transfer!, dual_to_interm_transfer!
export shared_matrix_load!, shared_vector_load!
export vector_load!, vector_write!

"""
Abstraction of shared memory layout for a matrix accessible both column and row-wise.
"""
struct DualAccessMatrix{T,D,V} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
    offset::Int32
end

"""
Constructor for DualAccessMatrix, for memory layout where one warp handles multiple matrices.
Meant for smaller matrices.
"""
function DualAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, warp_matrix_id::Int32, ::Val{:small}
) where {T,D}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32

    n_mats_per_warp = 32i32 ÷ D
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + padding
    offset = (wid - 1i32) * D * stride - stride - n_mats_per_warp + warp_matrix_id

    return DualAccessMatrix{T,D,Val{:small}}(shmem, offset)
end

"""
Constructor for DualAccessMatrix, for memory layout where one matrix is handled by D^2
threads. Meant for larger matrices, up to D=32.
"""
function DualAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, block_mtrx_id::Int32, ::Val{:large}
) where {T,D}
    offset = (block_mtrx_id - 1i32) * D * D
    return DualAccessMatrix{T,D,Val{:large}}(shmem, offset)
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
    A::DualAccessMatrix{T,D,Val{:small}}, i::Int32, j::Int32
) where {T,D}
    stride = _compute_stride(Val(D))
    n_mats_per_warp = _compute_n_mats_per_warp(Val(D))
    return A.shmem[A.offset + j * stride + i * n_mats_per_warp]
end

"""
Set index method for memory layout where one warp handles multiple matrices
"""
Base.@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrix{T,D,Val{:small}}, v::T, i::Int32, j::Int32
) where {T,D}
    stride = _compute_stride(Val(D))
    n_mats_per_warp = _compute_n_mats_per_warp(Val(D))
    return A.shmem[A.offset + j * stride + i * n_mats_per_warp] = v
end

"""
Get index method for memory layout where one matrix is handled by D^2 threads.
"""
Base.@propagate_inbounds @inline function Base.getindex(
    A::DualAccessMatrix{T,D,Val{:large}}, i::Int32, j::Int32
) where {T,D}
    interm_pad_freq = div(32i32, D & -D) * D
    raw_idx = A.offset + (j - 1) * D + i
    padded_amount = (raw_idx - 1i32) ÷ interm_pad_freq

    return A.shmem[raw_idx + padded_amount]
end

"""
Set index method for memory layout where one matrix is handled by D^2 threads.
"""
Base.@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrix{T,D,Val{:large}}, v::T, i::Int32, j::Int32
) where {T,D}
    interm_pad_freq = div(32i32, D & -D) * D
    raw_idx = A.offset + (j - 1) * D + i
    padded_amount = (raw_idx - 1i32) ÷ interm_pad_freq

    return A.shmem[raw_idx + padded_amount] = v
end

# Wrappers to handle Int32 case
@propagate_inbounds Base.getindex(A::AdjOrTransAbsMat{T}, i::Int32, j::Int32) where {T} =
    wrapperop(A)(A.parent[j, i])::T

# Support regular Int indexing (needed for Adjoint and other wrappers)
@propagate_inbounds @inline function Base.getindex(
    A::DualAccessMatrix{T,D,V}, i::Int, j::Int
) where {T,D,V}
    return getindex(A, Int32(i), Int32(j))
end

@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrix{T,D,V}, v::T, i::Int, j::Int
) where {T,D,V}
    return setindex!(A, v, Int32(i), Int32(j))
end

@inline Base.size(::DualAccessMatrix{T,D,V}) where {T,D,V} = (D, D)
@inline Base.length(::DualAccessMatrix{T,D,V}) where {T,D,V} = D * D
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

        if grid_vec_load <= N
            dest_idx = raw_idx
            src_idx = global_offset + raw_idx

            shmem[dest_idx] = global_arr[src_idx]
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

        if grid_vec_store <= N
            src_idx = raw_idx
            dest_idx = global_offset + raw_idx

            global_arr[dest_idx] = shmem[src_idx]
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

struct SharedMatrix{T,D,pad_interval} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
end

"""
A matrix that is shared across batches and backed by shared memory.

Since there is only one matrix, both rows and columns can be accessed in parallel using the
usual single access padding.
"""
function SharedMatrix(shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}) where {T,D}
    pad_interval = div(32i32, D & -D) * D
    return SharedMatrix{T,D,pad_interval}(shmem)
end

@propagate_inbounds @inline function Base.getindex(
    A::SharedMatrix{T,D,pad_interval}, i::Int32, j::Int32
) where {T,D,pad_interval}
    raw_idx = (j - 1i32) * D + i
    padding = (raw_idx - 1i32) ÷ pad_interval
    return A.shmem[raw_idx + padding]
end
@propagate_inbounds @inline function Base.setindex!(
    A::SharedMatrix{T,D,pad_interval}, v::T, i::Int32, j::Int32
) where {T,D,pad_interval}
    raw_idx = (j - 1i32) * D + i
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
abstract type DualAccessMatrixWrapper{T,D,V} <: AbstractMatrix{T} end

Base.parent(A::DualAccessMatrixWrapper{T,D,V}) where {T,D,V} = A.parent
Base.size(A::DualAccessMatrixWrapper{T,D,V}) where {T,D,V} = size(parent(A))

"""
Default getters and setters, no-ops
"""
@inline function wrapper_get(
    A::DualAccessMatrixWrapper{T,D,V}, v::T, i::Int32, j::Int32,
) where {T,D,V}
    return v
end

@inline function wrapper_set(
    A::DualAccessMatrixWrapper{T,D,V}, v::T, i::Int32, j::Int32,
) where {T,D,V}
    return v
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::DualAccessMatrixWrapper{T,D,V}, i::Int32, j::Int32,
) where {T,D,V}
    return wrapper_get(A, parent(A)[i, j], i, j)
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrixWrapper{T,D,V}, v::T, i::Int32, j::Int32,
) where {T,D,V}
    return parent(A)[i, j] = wrapper_set(A, v, i, j)
end

"""
Wrapper op for I - A
"""
struct IMinusSetterMatrix{T,D,V} <: DualAccessMatrixWrapper{T,D,V}
    parent::DualAccessMatrix{T,D,V}
end

Base.@propagate_inbounds @inline function wrapper_set(
    A::IMinusSetterMatrix{T,D,V}, v::T, i::Int32, j::Int32,
) where {T,D,V}
    return (i == j) * one(T) - v
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
    shmem, global_arr, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32  # Works if nthreads is divisible by 32, otherwise needs to be cld(nthreads, 32)
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = (n_mats_per_warp * D + padding) * D  # Shared memory covered by one warp
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
            )  # div(raw_idx - 1, warp_shmem_elem) != wid
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
    shmem, global_arr, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{:lower},
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32  # Works if nthreads is divisible by 32, otherwise needs to be cld(nthreads, 32)
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = (n_mats_per_warp * D + padding) * D  # Shared memory covered by one warp
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

                raw_idx_sym = (mod1(raw_mtrx, n_mats_per_warp) - 1i32) * D * D + j + (i - 1i32) * D

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
the warp was responsible for in previous calculations.
"""
@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{:indep},
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
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
            grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block

            if raw_mtrx <= n_mats_per_block &&
                grid_mtrx_load <= N &&
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
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{:indep}, ::Val{:lower}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
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
            grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block

            i = mod1(offset + lid, D)
            j = (raw_idx - 1i32 - (raw_mtrx - 1i32) * D * D) ÷ D + 1i32

            if raw_mtrx <= n_mats_per_block &&
                grid_mtrx_load <= N &&
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
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{:conseq}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    n_elements_per_block = n_mats_per_block * D * D

    tid = threadIdx().x
    bid = blockIdx().x

    sync_threads()

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
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
    shmem_dual, shmem_interm, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + padding
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
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
                warp_matrix_id
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
    shmem_interm, shmem_dual, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + padding
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
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
                warp_matrix_id
            )

            # Load from intermediate layout to dual-access layout
            shmem_interm[padded_idx_interm] = shmem_dual[padded_idx_dual]
        end
    end

    return nothing
end

@inline function _get_large_n_mats_per_block(
    ::Val{D}, ::Val{nthreads}, ::Val{:conseq}
) where {D,nthreads}
    return nthreads ÷ (D * D)
end

@inline function _get_large_n_mats_per_block(
    ::Val{D}, ::Val{nthreads}, ::Val{:indep}
) where {D,nthreads}
    return 1i32
end

"""
Function for loading matrices from global memory to shared memory, for the case
where one matrix is handled by D^2 threads. Meant for large matrices up to D=32.
"""
@inline function intermediate_layout_load!(
    shmem, global_arr, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:large}, ::Val{mode}
) where {D,nthreads,mode}
    tid = threadIdx().x
    bid = blockIdx().x

    n_mats_per_block = _get_large_n_mats_per_block(Val(D), Val(nthreads), Val(mode))

    interm_pad_freq = div(32i32, D & -D) * D
    padded_amount_per_block = (n_mats_per_block * D * D - 1i32) ÷ interm_pad_freq
    n_elements_per_block = D * D * n_mats_per_block + padded_amount_per_block

    base_addr = (bid - 1i32) * n_mats_per_block * D * D + 1i32
    align_offset = (base_addr - 1i32) % 32i32

    offset = 0i32

    while offset < n_elements_per_block + align_offset
        raw_idx = offset + tid - align_offset
        raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32
        grid_mtrx_load = (bid - 1i32) * n_mats_per_block + raw_mtrx

        if raw_mtrx <= n_mats_per_block && grid_mtrx_load <= N && raw_idx > 0i32
            padded_amount = (raw_idx - 1i32) ÷ interm_pad_freq

            src_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
            dest_idx = raw_idx + padded_amount

            @inbounds shmem[dest_idx] = global_arr[src_idx]
        end

        offset += nthreads
    end

    return nothing
end

"""
Function for writing matrices from shared memory to global memory, for the case
where one matrix is handled by D^2 threads. Meant for large matrices up to D=32.
"""
@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:large}, ::Val{mode}
) where {D,nthreads,mode}
    tid = threadIdx().x
    bid = blockIdx().x

    n_mats_per_block = _get_large_n_mats_per_block(Val(D), Val(nthreads), Val(mode))

    interm_pad_freq = div(32i32, D & -D) * D
    padded_amount_per_block = (n_mats_per_block * D * D - 1i32) ÷ interm_pad_freq
    n_elements_per_block = D * D * n_mats_per_block + padded_amount_per_block

    base_addr = (bid - 1i32) * n_mats_per_block * D * D + 1i32
    align_offset = (base_addr - 1i32) % 32i32

    offset = 0i32

    while offset < n_elements_per_block + align_offset
        raw_idx = offset + tid - align_offset
        raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32
        grid_mtrx_load = (bid - 1i32) * n_mats_per_block + raw_mtrx

        if raw_mtrx <= n_mats_per_block && grid_mtrx_load <= N && raw_idx > 0i32
            padded_amount = (raw_idx - 1i32) ÷ interm_pad_freq

            dest_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
            src_idx = raw_idx + padded_amount

            @inbounds global_arr[dest_idx] = shmem[src_idx]
        end

        offset += nthreads
    end

    return nothing
end
