import Base: @propagate_inbounds
import LinearAlgebra: AdjOrTransAbsMat, wrapperop

export DualAccessMatrix, SingleAccessMatrix
export intermediate_layout_load!, intermediate_layout_write!
export interm_to_dual_transfer!, dual_to_interm_transfer!

# TODO: each of these have their own offset which is wasteful of registers
# Could use a pointer to a shared index information struct but will that use a register too?
struct DualAccessMatrix{T,D} <: AbstractMatrix{T}
    shmem::CuDeviceVector{T,CUDA.AS.Shared}
    offset::Int32
end

function DualAccessMatrix(
    shmem::CuDeviceVector{T,CUDA.AS.Shared}, ::Val{D}, wid::Int32, warp_matrix_id::Int32
) where {T,D}
    n_mats_per_warp = 32i32 ÷ D
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + padding
    offset = (wid - 1i32) * D * stride - stride - n_mats_per_warp + warp_matrix_id
    return DualAccessMatrix{T,D}(shmem, offset)
end

@inline function _compute_stride(::Val{D}) where {D}
    n_mats_per_warp = 32i32 ÷ D
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    return n_mats_per_warp * D + padding
end

@inline function _compute_n_mats_per_warp(::Val{D}) where {D}
    return 32i32 ÷ D
end

Base.@propagate_inbounds @inline function Base.getindex(
    A::DualAccessMatrix{T,D}, i::Int32, j::Int32
) where {T,D}
    stride = _compute_stride(Val(D))
    n_mats_per_warp = _compute_n_mats_per_warp(Val(D))
    return A.shmem[A.offset + j * stride + i * n_mats_per_warp]
end

Base.@propagate_inbounds @inline function Base.setindex!(
    A::DualAccessMatrix{T,D}, v::T, i::Int32, j::Int32
) where {T,D}
    stride = _compute_stride(Val(D))
    n_mats_per_warp = _compute_n_mats_per_warp(Val(D))
    return A.shmem[A.offset + j * stride + i * n_mats_per_warp] = v
end

# Wrappers to handle Int32 case
@propagate_inbounds Base.getindex(A::AdjOrTransAbsMat{T}, i::Int32, j::Int32) where {T} =
    wrapperop(A)(A.parent[j, i])::T

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
    return A.shmem[A.outer_offset + warp_idx + padding]
end
@propagate_inbounds @inline function Base.setindex!(
    A::SingleAccessMatrix{T,pad_interval}, v::T, i::Int32
) where {T,pad_interval}
    warp_idx = A.inner_offset + i
    padding = (warp_idx - 1i32) ÷ pad_interval
    return A.shmem[A.outer_offset + warp_idx + padding] = v
end

#####################################
#### EXPLICIT MEMORY SUB-KERNELS ####
#####################################

@inline function intermediate_layout_load!(
    shmem, global_arr, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32  # Works if nthreads is divisible by 32, otherwise needs to be cld(nthreads, 32)
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    if lid > active_lanes
        return nothing
    end

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = (n_mats_per_warp * D + padding) * D  # Shared memory covered by one warp
    warp_shmem_elem = n_mats_per_warp * D * D  # Shared memory used my one warp (excluding padding for dual)
    interm_pad_freq = (D & (D - 1i32)) == 0i32 ? 32i32 : (warp_shmem_elem + 1i32)  # No padding needed if D is not a power of 2
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32  # Global start idx for global memory
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32  # Global start for shared memory

    # Each thread loads warp_shmem_elems starting from start
    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
            raw_idx = start_raw + offset + lid - 1i32  # How many-th element to load in the block: [1, n_elements_per_block]
            raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32  # How many-th matrix to load: [1, n_mats_per_block]
            grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block  # How many-th global matrix 1 ... N to load

            if raw_mtrx <= n_mats_per_block && grid_mtrx_load <= N
                padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

                src_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
                dest_idx = start_offset + offset + lid - 1i32 + padded_amount

                shmem[dest_idx] = global_arr[src_idx]
            end

            offset += active_lanes
        end
    end

    return nothing
end

@inline function intermediate_layout_write!(
    global_arr, shmem, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    if lid > active_lanes
        return nothing
    end

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
    warp_shmem_elem = n_mats_per_warp * D * D
    interm_pad_freq = (D & (D - 1i32)) == 0i32 ? 32i32 : (warp_shmem_elem + 1i32)
    start_raw = (wid - 1i32) * warp_shmem_elem + 1i32
    start_offset = (wid - 1i32) * warp_shmem_size + 1i32

    # Each thread loads warp_shmem_elems starting from start
    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
            raw_idx = start_raw + offset + lid - 1i32
            raw_mtrx = div(raw_idx - 1i32, D * D) + 1i32
            grid_mtrx_load = raw_mtrx + (bid - 1i32) * n_mats_per_block

            if raw_mtrx <= n_mats_per_block && grid_mtrx_load <= N
                padded_amount = (offset + lid - 1i32) ÷ interm_pad_freq

                dest_idx = (bid - 1i32) * n_mats_per_block * D * D + raw_idx
                src_idx = start_offset + offset + lid - 1i32 + padded_amount

                global_arr[dest_idx] = shmem[src_idx]
            end

            offset += active_lanes
        end
    end

    return nothing
end

@inline function interm_to_dual_transfer!(
    shmem_dual, shmem_interm, ::Val{D}, ::Val{n_threads}, N::Int32
) where {D,n_threads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = n_threads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + padding
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
    warp_shmem_elem = n_mats_per_warp * D * D
    interm_pad_freq = (D & (D - 1i32)) == 0i32 ? 32i32 : (warp_shmem_elem + 1i32)

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    matrix_thread = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    @inbounds if lid <= active_lanes && grid_matrix_id <= N
        col = matrix_thread

        # Loop over the rows of each matrix
        for row in 1i32:D
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

@inline function dual_to_interm_transfer!(
    shmem_interm, shmem_dual, ::Val{D}, ::Val{n_threads}, N::Int32
) where {D,n_threads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = n_threads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    active_lanes = n_mats_per_warp * D

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)

    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    stride = n_mats_per_warp * D + padding
    warp_shmem_size = (n_mats_per_warp * D + padding) * D
    warp_shmem_elem = n_mats_per_warp * D * D
    interm_pad_freq = (D & (D - 1i32)) == 0i32 ? 32i32 : (warp_shmem_elem + 1i32)

    warp_matrix_id = div(lid - 1i32, D) + 1i32
    matrix_thread = mod1(lid, D)
    block_matrix_id = (wid - 1i32) * n_mats_per_warp + warp_matrix_id
    grid_matrix_id = (bid - 1i32) * n_mats_per_block + block_matrix_id

    @inbounds if lid <= active_lanes && grid_matrix_id <= N
        col = matrix_thread

        # Loop over the rows of each matrix
        for row in 1i32:D
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
