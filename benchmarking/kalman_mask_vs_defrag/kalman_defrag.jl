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

@inline function kernel_kalman_defrag!(
    P_out,
    P_in,
    F_global,
    Q_global,
    H_global,
    R_global,
    ::Val{Dx},
    ::Val{Dy},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
) where {Dx,Dy,D,nthreads}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32

    pad_interval_xx = div(32i32, Dx & -Dx) * Dx
    shmem_fixed_size_xx = Dx * Dx + (Dx * Dx - 1i32) ÷ pad_interval_xx

    pad_interval_yy = div(32i32, Dy & -Dy) * Dy
    shmem_fixed_size_yy = Dy * Dy + (Dy * Dy - 1i32) ÷ pad_interval_yy

    pad_interval_yx = div(32i32, Dy & -Dy) * Dy
    shmem_fixed_size_yx = Dy * Dx + (Dy * Dx - 1i32) ÷ pad_interval_yx

    shmem_F = CuStaticSharedArray(Float32, (shmem_fixed_size_xx,))
    shmem_Q = CuStaticSharedArray(Float32, (shmem_fixed_size_xx,))
    shmem_H = CuStaticSharedArray(Float32, (shmem_fixed_size_yx,))
    shmem_R = CuStaticSharedArray(Float32, (shmem_fixed_size_yy,))

    if wid == 1i32
        shared_matrix_load!(shmem_F, F_global, Val(Dx), Val(Dx))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_Q, Q_global, Val(Dx), Val(Dx))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_global, Val(Dy), Val(Dx))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_R, R_global, Val(Dy), Val(Dy))
    end

    F = SharedMatrix(shmem_F, Val(Dx), Val(Dx))
    Q = SharedMatrix(shmem_Q, Val(Dx), Val(Dx))
    H = SharedMatrix(shmem_H, Val(Dy), Val(Dx))
    R = SharedMatrix(shmem_R, Val(Dy), Val(Dy))

    (shmem_elems, d, warp_matrix_id, block_mtrx_id, grid_mtrx_id, n_mats_per_warp, n_mats_per_block) = get_shmem_elems(Val(D), Val(D), Val(D), Val(nthreads))
    (shmem_elems_xx, d_xx, warp_matrix_id_xx, block_mtrx_id_xx, grid_mtrx_id_xx, n_mats_per_warp_xx, n_mats_per_block_xx) = get_shmem_elems(Val(Dx), Val(Dx), Val(D), Val(nthreads))
    (shmem_elems_yy, d_yy, warp_matrix_id_yy, block_mtrx_id_yy, grid_mtrx_id_yy, n_mats_per_warp_yy, n_mats_per_block_yy) = get_shmem_elems(Val(Dy), Val(Dy), Val(D), Val(nthreads))

    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_4 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load P shape: (Dx,Dx)
    intermediate_layout_load!(shmem_1, P_in, Val(Dx), Val(Dx), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_1, Val(Dx), Val(Dx), Val(D), Val(nthreads), N, Val(:small))

    sync_threads()

    warps_active = cld(n_mats_per_block, n_mats_per_warp_xx)
    if warp_matrix_id_xx <= n_mats_per_warp_xx && grid_mtrx_id_xx <= N && wid <= warps_active && block_mtrx_id_xx <= n_mats_per_block
        # Calculating which warp and how many-th matrix within the warp the current matrix belongs
        # to under the new distribution
        wid_retrieve = (block_mtrx_id_xx - 1i32) ÷ n_mats_per_warp + 1i32
        warp_matrix_id_retrieve = mod1(block_mtrx_id_xx, n_mats_per_warp)
        
        M1 = DualAccessMatrix(shmem_1, Val(Dx), warp_matrix_id_xx, Val(:small))
        M3 = DualAccessMatrix(shmem_3, Val(Dx), warp_matrix_id_xx, Val(:small))

        # Transfer P (D,D) -> (Dx,Dx)
        P_D = DualAccessMatrix(shmem_2, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        P_pred = DualAccessMatrix(shmem_4, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))

        P_Dx = P_D

        # Compute P_pred = FPF' + Q
        # NOTE: The following operations cannot mix memory slots between two different layouts due to race conditions
        batch_op!(*, M3, F, P_Dx, d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))
        batch_op!(*, M1, M3, F', d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))
        batch_op!(+, P_pred, M1, Q, d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))
    end

    sync_threads()

    # Compute H P_pred H' normally wihtin (D,D)
    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        HPH_trans = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        P_pred = DualAccessMatrix(shmem_4, Val(D), warp_matrix_id, Val(:small))
        HP = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        batch_op!(*, HP, H, P_pred, d, Val(Dy), Val(Dx), Val(Dx), Val(:small))
        batch_op!(*, HPH_trans, HP, H', d, Val(Dy), Val(Dx), Val(Dy), Val(:small))
    end

    sync_threads()

    # Compute S = (H P_pred H') + Q and K = P_pred H' S^{-1}
    warps_active = cld(n_mats_per_block, n_mats_per_warp_yy)
    if warp_matrix_id_yy <= n_mats_per_warp_yy && grid_mtrx_id_yy <= N && wid <= warps_active && block_mtrx_id_yy <= n_mats_per_block
        wid_retrieve = (block_mtrx_id_yy - 1i32) ÷ n_mats_per_warp + 1i32
        warp_matrix_id_retrieve = mod1(block_mtrx_id_yy, n_mats_per_warp)
        
        HPH_trans = DualAccessMatrix(shmem_1, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        S = DualAccessMatrix(shmem_2, Val(Dy), warp_matrix_id_yy, Val(:small))
        U = HPH_trans
        
        # S = HPH' + R
        batch_op!(+, S, HPH_trans, R, d_yy, Val(Dy), Val(Dy), Val(Dy), Val(:small))

        # Cholesky of S into M1 (D,D)
        batch_op!(cholesky, U, S, d_yy, Val(Dy), Val(Dy), warp_matrix_id_yy, Val(:small))
    end

    sync_threads()

    # Compute K = HP / S normally within (D,D)
    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        U = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        HP = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))
        K_trans = HP

        # Compute U' \ HP
        batch_op!(\, HP, LowerTriangular(U'), HP, d, Val(Dy), Val(Dx), Val(D), Val(:small))

        # Compute K = U \ (U' \ HP) to obtain K = HP / S
        batch_op!(\, K_trans, UpperTriangular(U), HP, d, Val(Dy), Val(Dx), Val(D), Val(:small))
    end

    sync_threads()

    # Compute rest in (Dx,Dx)
    warps_active = cld(n_mats_per_block, n_mats_per_warp_xx)
    if warp_matrix_id_xx <= n_mats_per_warp_xx && grid_mtrx_id_xx <= N && wid <= warps_active && block_mtrx_id_xx <= n_mats_per_block
        # Calculating which warp and how many-th matrix within the warp the current matrix belongs
        # to under the new distribution
        wid_retrieve = (block_mtrx_id_xx - 1i32) ÷ n_mats_per_warp + 1i32
        warp_matrix_id_retrieve = mod1(block_mtrx_id_xx, n_mats_per_warp)
        
        M1 = DualAccessMatrix(shmem_1, Val(Dx), warp_matrix_id_xx, Val(:small))
        M2 = DualAccessMatrix(shmem_2, Val(Dx), warp_matrix_id_xx, Val(:small))
        K_trans = DualAccessMatrix(shmem_3, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        P_pred = DualAccessMatrix(shmem_4, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))

        # M1 <- (I - KH)
        batch_op!(*, IAddSubSetterMatrix(M1, 1.0f0, -1.0f0), K_trans', H, d_xx, Val(Dx), Val(Dy), Val(Dx), Val(:small))

        # M2 <- (I - KH) * P_pred
        batch_op!(*, M2, M1, P_pred, d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))
    end

    sync_threads()

    if wid <= warps_active
        M = DualAccessMatrix(shmem_2, Val(Dx), warp_matrix_id_xx, Val(:small))
        dual_to_interm_transfer!(shmem_3, M, Val(Dx), Val(Dx), Val(Dx), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
        intermediate_layout_write!(P_out, shmem_3, Val(Dx), Val(Dx), Val(Dx), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
    end

    return nothing
end