@inline function get_shmem_elems(::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}) where {D1,D2,D,nthreads}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32

    n_mats_per_warp = 32i32 ÷ D1
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D1, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)

    warp_matrix_id = div(lid - 1i32, D1) + 1i32
    block_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp
    d = mod1(lid, D1)

    n_mats_per_warp_global = 32i32 ÷ D
    n_mats_per_block_global = n_warps * n_mats_per_warp_global
    grid_mtrx_id = block_mtrx_id + (bid - 1i32) * n_mats_per_block_global

    warp_shmem_size = n_mats_per_warp * D1 * D2 + dual_padding * (D2 - 1i32)
    shmem_elems = warp_shmem_size * n_warps

    return shmem_elems, d, warp_matrix_id, block_mtrx_id, grid_mtrx_id, n_mats_per_warp, n_mats_per_block
end

@inline function kernel_mul_defrag!(
    M_out,
    M_in,
    A_global,
    ::Val{D1},
    ::Val{D},
    ::Val{nthreads},
    ::Val{n_muls},
    N::Int32,
) where {D1,D,n_muls,nthreads}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32

    pad_interval = div(32i32, D1 & -D1) * D1
    shmem_fixed_size = D1 * D1 + (D1 * D1 - 1i32) ÷ pad_interval

    shmem_A = CuStaticSharedArray(Float32, (shmem_fixed_size,))

    if wid == 1i32
        shared_matrix_load!(shmem_A, A_global, Val(D1), Val(D1))
    end

    A = SharedMatrix(shmem_A, Val(D1), Val(D1))

    (shmem_elems, d, warp_matrix_id, block_mtrx_id, grid_mtrx_id, n_mats_per_warp, n_mats_per_block) = get_shmem_elems(Val(D), Val(D), Val(D), Val(nthreads))
    (shmem_elems_small, d_small, warp_matrix_id_small, block_mtrx_id_small, grid_mtrx_id_small, n_mats_per_warp_small, n_mats_per_block_small) = get_shmem_elems(Val(D1), Val(D1), Val(D), Val(nthreads))

    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load M_in shape: (D1,D1) into (D,D)
    intermediate_layout_load!(shmem_2, M_in, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

    sync_threads()

    warps_active = cld(n_mats_per_block, n_mats_per_warp_small)
    if warp_matrix_id_small <= n_mats_per_warp_small && grid_mtrx_id <= N && wid <= warps_active && block_mtrx_id_small <= n_mats_per_block
        # Calculating which warp and how many-th matrix within the warp the current matrix belongs
        # to under the new distribution
        wid_retrieve = (block_mtrx_id_small - 1i32) ÷ n_mats_per_warp + 1i32
        warp_matrix_id_retrieve = mod1(block_mtrx_id_small, n_mats_per_warp)
        
        M1 = DualAccessMatrix(shmem_1, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        M2 = DualAccessMatrix(shmem_2, Val(D1), warp_matrix_id_small, Val(:small))

        for _ in 1i32:n_muls
            batch_op!(*, M2, M1, A, d_small, Val(D1), Val(D1), Val(D1), Val(:small))
        end
    end

    sync_threads()

    if wid <= warps_active
        M = DualAccessMatrix(shmem_2, Val(D1), warp_matrix_id_small, Val(:small))
        dual_to_interm_transfer!(shmem_1, M, Val(D1), Val(D1), Val(D1), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
        intermediate_layout_write!(M_out, shmem_1, Val(D1), Val(D1), Val(D1), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
    end

    return nothing
end