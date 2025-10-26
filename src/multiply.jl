export kernel_matmul!

function kernel_matmul_n_mats_per_warp!(
    Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load A
    intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N)

    # Create dual-access matrices
    A = DualAccessMatrix(shmem_1, Val(D), wid, warp_matrix_id)
    B = DualAccessMatrix(shmem_2, Val(D), wid, warp_matrix_id)
    C = DualAccessMatrix(shmem_3, Val(D), wid, warp_matrix_id)

    # Perform operation with optional adjoints
    batch_op!(*, C, A, B, d, Val(D))

    # Store C
    dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)
    intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N)

    return nothing
end

function kernel_matmul_d2_threads_per_mat!(
    Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D, nthreads}
    tid = threadIdx().x
    bid = blockIdx().x

    n_mats_per_block = nthreads ÷ (D * D)
    interm_pad_freq = div(32i32, D & -D) * D
    padded_amount_per_mat = (D * D - 1i32) ÷ interm_pad_freq
    mat_shmem_size = D * D + padded_amount_per_mat
    block_mtrx_id = div(tid - 1i32, D * D) + 1i32
    grid_mtrx_load = (bid - 1i32) * n_mats_per_block + block_mtrx_id

    shmem_elems = mat_shmem_size * n_mats_per_block

    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    intermediate_layout_load!(shmem_1, As, Val(D), Val(nthreads), N)
    intermediate_layout_load!(shmem_2, Bs, Val(D), Val(nthreads), N)

    sync_threads()
    if block_mtrx_id <= n_mats_per_block && grid_mtrx_load <= N
        A = DualAccessMatrix(shmem_1, Val(D), block_mtrx_id)
        B = DualAccessMatrix(shmem_2, Val(D), block_mtrx_id)
        C = DualAccessMatrix(shmem_3, Val(D), block_mtrx_id)
        batch_op!(*, C, A, B, Val(D))
    end
    sync_threads()

    intermediate_layout_write!(Cs, shmem_3, Val(D), Val(nthreads), N)

    return nothing
end

function kernel_matmul_one_mats_per_warp!(
    Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32
) where {D,nthreads}
    n_mats_per_warp = 1i32
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    interm_pad_freq = div(32i32, D & -D) * D

    padded_amount_per_warp = (D * D) ÷ interm_pad_freq
    warp_shmem_size = D * D + padded_amount_per_warp

    tid = threadIdx().x
    lid = mod1(tid, 32i32)

    block_matrix_id = div(tid - 1i32, 32i32) + 1i32
    d = mod1(lid, D)

    shmem_elems = warp_shmem_size * n_mats_per_block
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    intermediate_layout_load!(shmem_1, As, Val(D), Val(nthreads), N)
    intermediate_layout_load!(shmem_2, Bs, Val(D), Val(nthreads), N)

    A = DualAccessMatrix(shmem_1, Val(D), block_matrix_id)
    B = DualAccessMatrix(shmem_2, Val(D), block_matrix_id)
    C = DualAccessMatrix(shmem_3, Val(D), block_matrix_id)

    if block_matrix_id <= n_mats_per_block && lid <= D
        batch_op!(*, C, A, B, d, Val(D))
    end

    intermediate_layout_write!(Cs, shmem_3, Val(D), Val(nthreads), N)

    return nothing
end

@static if VERSION === :NMatsPerWarp
    kernel_matmul! = kernel_matmul_n_mats_per_warp!
elseif VERSION === :D2ThreadsPerMat
    kernel_matmul! = kernel_matmul_d2_threads_per_mat!
elseif VERSION === :OneMatPerWarp
    kernel_matmul! = kernel_matmul_one_mats_per_warp!
end