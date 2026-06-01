using CUDA
using CUDA: i32

#####################################
#### EXPLICIT MEMORY SUB-KERNELS ####
#####################################


@inline function intermediate_layout_load_untuned!(
    shmem, global_arr, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small},
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

    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
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

            offset += 32i32
        end
    end

    return nothing
end

@inline function interm_to_dual_transfer_untuned!(
    shmem_dual, shmem_interm, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small},
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

    @inbounds if lid <= active_lanes && grid_matrix_id <= N && col <= D2
        for row in (1i32):D1
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

@inline function dual_to_interm_transfer_untuned!(
    shmem_interm, M_dual, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small},
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

    @inbounds if warp_matrix_id <= n_mats_per_warp && grid_matrix_id <= N && col <= D2
        for row in (1i32):D1
            logical_idx = (warp_matrix_id - 1i32) * D1 * D2 + (col - 1i32) * D1 + row
            padding = (logical_idx - 1i32) ÷ interm_pad_freq
            padded_idx_interm = logical_idx + padding + (wid - 1i32) * warp_shmem_size

            shmem_interm[padded_idx_interm] = M_dual[row, col]
        end
    end

    return nothing
end

@inline function intermediate_layout_write_untuned!(
    global_arr, shmem, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small},
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

    @inbounds begin
        offset = 0i32
        while offset < warp_shmem_elem
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

            offset += 32i32
        end
    end

    return nothing
end
