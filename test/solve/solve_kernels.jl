@inline function kernel_backward_solve!(
    Cs, Us, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load U
    intermediate_layout_load!(shmem_3, Us, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        # Create dual-access matrices
        U_mat = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        # Perform backward solve: C = U \ B
        U = UpperTriangular(U_mat)
        batch_op!(\, C, U, B, d, Val(D), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N, Val(:small), Val(mode))

    return nothing
end

@inline function kernel_backward_solve!(
    Cs, Us, Bs, ::Val{D1}, ::Val{D2}, ::Val{nthreads}, N::Int32, ::Val{:small},
) where {D1,D2,nthreads}
    D = max(D1, D2)
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load U
    intermediate_layout_load!(shmem_3, Us, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    # Create dual-access matrices
    U_mat = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
    C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        # Perform backward solve: C = U \ B
        U = UpperTriangular(U_mat)
        batch_op!(\, C, U, B, d, Val(D1), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

@inline function kernel_forward_solve!(
    Cs, Ls, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load L
    intermediate_layout_load!(shmem_3, Ls, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        # Create dual-access matrices
        L_mat = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        # Perform forward solve: C = L \ B
        L = LowerTriangular(L_mat)
        batch_op!(\, C, L, B, d, Val(D), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N, Val(:small), Val(mode))

    return nothing
end

@inline function kernel_forward_solve!(
    Cs, Ls, Bs, ::Val{D1}, ::Val{D2}, ::Val{nthreads}, N::Int32, ::Val{:small},
) where {D1,D2,nthreads}
    D = max(D1, D2)
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load U
    intermediate_layout_load!(shmem_3, Ls, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    # Create dual-access matrices
    L = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
    C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        # Perform backward solve: C = U \ B
        batch_op!(\, C, LowerTriangular(L), B, d, Val(D1), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end