export kernel_matmul!

"""
Matrix multiplication kernel for the case where one warp handles multiple matrices.
"""
function kernel_matmul!(
    Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load A
    intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp
        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        batch_op!(*, C, A, B, d, Val(D), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N, Val(:small), Val(mode))

    return nothing
end

"""
Matrix multiplication kernel for the case where one matrix is handled by D^2 threads.
"""
function kernel_matmul!(
    Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:large}, ::Val{mode},
) where {D, nthreads,mode}
    tid = threadIdx().x
    bid = blockIdx().x

    n_mats_per_block = nthreads ÷ (D * D)
    interm_pad_freq = div(32i32, D & -D) * D
    block_mtrx_id = div(tid - 1i32, D * D) + 1i32
    grid_mtrx_load = (bid - 1i32) * n_mats_per_block + block_mtrx_id

    padded_amount_per_block = (n_mats_per_block * D * D - 1i32) ÷ interm_pad_freq
    shmem_elems = D * D * n_mats_per_block + padded_amount_per_block

    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    intermediate_layout_load!(shmem_1, As, Val(D), Val(nthreads), N, Val(:large))
    intermediate_layout_load!(shmem_2, Bs, Val(D), Val(nthreads), N, Val(:large))

    sync_threads()
    if block_mtrx_id <= n_mats_per_block && grid_mtrx_load <= N
        A = DualAccessMatrix(shmem_1, Val(D), block_mtrx_id, Val(:large))
        B = DualAccessMatrix(shmem_2, Val(D), block_mtrx_id, Val(:large))
        C = DualAccessMatrix(shmem_3, Val(D), block_mtrx_id, Val(:large))
        batch_op!(*, C, A, B, Val(D), Val(:large), Val(mode))
    end
    sync_threads()

    intermediate_layout_write!(Cs, shmem_3, Val(D), Val(nthreads), N, Val(:large))

    return nothing
end

function kernel_matmul!(
    Cs, As, Bs, A_adj, B_adj, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load A
    intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp
        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        # Perform operation with optional adjoints
        A_mat = A_adj ? A' : A
        B_mat = B_adj ? B' : B
        batch_op!(*, C, A_mat, B_mat, d, Val(D), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N, Val(:small), Val(mode))

    return nothing
end

function kernel_matmul!(
    Cs, As, Bs, A_adj, B_adj, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:large}, ::Val{mode},
) where {D,nthreads,mode}
    tid = threadIdx().x
    bid = blockIdx().x

    n_mats_per_block = nthreads ÷ (D * D)
    interm_pad_freq = div(32i32, D & -D) * D
    block_mtrx_id = div(tid - 1i32, D * D) + 1i32
    grid_mtrx_load = (bid - 1i32) * n_mats_per_block + block_mtrx_id

    padded_amount_per_block = (n_mats_per_block * D * D - 1i32) ÷ interm_pad_freq
    shmem_elems = D * D * n_mats_per_block + padded_amount_per_block

    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    intermediate_layout_load!(shmem_1, As, Val(D), Val(nthreads), N, Val(:large))
    intermediate_layout_load!(shmem_2, Bs, Val(D), Val(nthreads), N, Val(:large))

    sync_threads()
    if block_mtrx_id <= n_mats_per_block && grid_mtrx_load <= N
        A = DualAccessMatrix(shmem_1, Val(D), block_mtrx_id, Val(:large))
        B = DualAccessMatrix(shmem_2, Val(D), block_mtrx_id, Val(:large))
        C = DualAccessMatrix(shmem_3, Val(D), block_mtrx_id, Val(:large))

        A_mat = A_adj ? A' : A
        B_mat = B_adj ? B' : B
        batch_op!(*, C, A_mat, B_mat, Val(D), Val(:large), Val(mode))
    end
    sync_threads()

    intermediate_layout_write!(Cs, shmem_3, Val(D), Val(nthreads), N, Val(:large))

    return nothing
end