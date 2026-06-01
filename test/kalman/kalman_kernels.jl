using KernelAbstractions.Extras: @unroll

@inline function kernel_kalman!(
    Ps_out,
    Ps_in,
    A_global,
    Q_global,
    H_global,
    R_global,
    µ_out,
    µ_in,
    b_in,
    z_in,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
    ::Val{mode},
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    shmem_vec_elems = D * n_warps * n_mats_per_warp
    shmem_vec_1 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
    shmem_vec_2 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
    shmem_vec_3 = CuStaticSharedArray(Float32, (shmem_vec_elems,))

    pad_interval = div(32i32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_Q = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_H = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_R = CuStaticSharedArray(Float32, (shmem_size_fixed,))

    shmem_vec_size_fixed = D
    shmem_vec_b = CuStaticSharedArray(Float32, (shmem_vec_size_fixed,))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_global, Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_Q, Q_global, Val(D))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_global, Val(D))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_R, R_global, Val(D))
    end
    if wid == 5i32
        shared_vector_load!(shmem_vec_b, b_in, Val(D))
    end
    sync_threads()

    A = SharedMatrix(shmem_A, Val(D))
    Q = SharedMatrix(shmem_Q, Val(D))
    H = SharedMatrix(shmem_H, Val(D))
    R = SharedMatrix(shmem_R, Val(D))
    b = SharedVector(shmem_vec_b, Val(D))

    # Load P
    intermediate_layout_load!(shmem_1, Ps_in, Val(D), Val(nthreads), N, Val(:small))#, Val(:lower))
    interm_to_dual_transfer!(shmem_3, shmem_1, Val(D), Val(nthreads), N, Val(:small))

    # Load µ, z
    vector_load!(shmem_vec_1, µ_in, Val(D), Val(nthreads), N)
    vector_load!(shmem_vec_2, z_in, Val(D), Val(nthreads), N)

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        B3 = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))
        v1 = BatchedVector(shmem_vec_1, Val(D), warp_matrix_id)
        v2 = BatchedVector(shmem_vec_2, Val(D), warp_matrix_id)
        v3 = BatchedVector(shmem_vec_3, Val(D), warp_matrix_id)

        # ######################
        # #### PREDICT STEP ####
        # ######################

        batch_op!(*, B2, A, B3, d, Val(D), Val(:small))
        batch_op!(*, B1, B2, A', d, Val(D), Val(:small))
        batch_op!(+, B3, B1, Q, d, Val(D), Val(:small))
        # B3 now contains P_pred. We keep this until the final update step.

        # v3 <- A * µ
        batch_op!(*, v3, A, v1, d, Val(D), Val(:small))

        # v1 <- (A * µ) + b
        batch_op!(+, v1, v3, b, d, Val(D), Val(:small))
        # v1 now contains µ_{k|k-1}

        #####################
        #### KALMAN GAIN ####
        #####################

        # K = P_pred * H' / S where S = H * P_pred * H' + R
        # We compute K' = S^{-1} * H * P_pred, then K = K'

        # H * P_pred → B1
        batch_op!(*, B1, H, B3, d, Val(D), Val(:small))
        # B1 now contains H * P_pred = (P_pred * H')' since P_pred is symmetric

        # H * P_pred * H' → B2
        batch_op!(*, B2, B1, H', d, Val(D), Val(:small))

        # S = H*P_pred*H' + R → B2
        batch_op!(+, B2, B2, R, d, Val(D), Val(:small))

        # In-place Cholesky of S (B2 becomes U where S = U'*U)
        batch_op!(cholesky, B2, d, Val(D), n_mats_per_warp, warp_matrix_id, Val(:small))

        # In-place forward solve U' \ B1 → B1 (X = (U')^{-1} * H*P_pred)
        # We need U' for forward solve
        batch_op!(\, LowerTriangular(B2'), B1, d, Val(D), Val(:small))

        # Backward solve U \ B1 → B1 (K' = U^{-1} * X = S^{-1} * H * P_pred)
        batch_op!(\, UpperTriangular(B2), B1, d, Val(D), Val(:small))
        # B1 now contains K'

        #####################
        #### UPDATE STEP ####
        #####################

        # Use P_new = (I - K*H) * P_pred form of update

        # Transpose K' in B2 to get K = P_pred * H' / S
        batch_op!(*, IAddSubSetterMatrix(B2, 1.0f0, -1.0f0), B1', H, d, Val(D), Val(:small))
        # B2 now contains (I - K*H)

        # (I - K * H) * x → v3
        batch_op!(*, v3, B2, v1, d, Val(D), Val(:small))

        # K * z → v1
        batch_op!(*, v1, B1', v2, d, Val(D), Val(:small))

        # x_new = (I - K*H) * x + K * z → v1
        batch_op!(+, v1, v3, v1, d, Val(D), Val(:small))

        batch_op!(*, B1, B2, B3, d, Val(D), Val(:small))
        # B2 now contains P_new = (I - K*H) * P_pred
    end

    # Write P_new (final output) - B2/shmem_2 contains P_new
    dual_to_interm_transfer!(shmem_2, shmem_1, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(
        Ps_out, shmem_2, Val(D), Val(nthreads), N, Val(:small), Val(mode)#, Val(:lower),
    )

    # Writing µ_new
    vector_write!(µ_out, shmem_vec_1, Val(D), Val(nthreads), N)

    return nothing
end

@inline function kernel_kalman_cov!(
    Ps_out,
    Ps_in,
    A_global,
    Q_global,
    H_global,
    R_global,
    ::Val{D},
    ::Val{nthreads},
    n_steps::Int32,
    N::Int32,
    ::Val{:small},
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    shmem_3 = CuDynamicSharedArray(Float32, shmem_elems, 2 * shmem_elems * sizeof(Float32))

    pad_interval = div(32i32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32))
    shmem_Q = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32) + shmem_size_fixed * sizeof(Float32))
    shmem_H = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32) + 2 * shmem_size_fixed * sizeof(Float32))
    shmem_R = CuDynamicSharedArray(Float32, shmem_size_fixed, 3 * shmem_elems * sizeof(Float32) + 3 * shmem_size_fixed * sizeof(Float32))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_global, Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_Q, Q_global, Val(D))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_global, Val(D))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_R, R_global, Val(D))
    end
    sync_threads()

    A = SharedMatrix(shmem_A, Val(D))
    Q = SharedMatrix(shmem_Q, Val(D))
    H = SharedMatrix(shmem_H, Val(D))
    R = SharedMatrix(shmem_R, Val(D))

    # Load P
    intermediate_layout_load!(shmem_1, Ps_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_3, shmem_1, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    sync_warp()

    B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
    B3 = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        for _ in 1i32:n_steps
            ######################
            #### PREDICT STEP ####
            ######################

            batch_op!(*, B2, A, B3, d, Val(D), Val(D), Val(D), Val(:small))
            batch_op!(*, B3, B2, A', d, Val(D), Val(D), Val(D), Val(:small))
            batch_op!(+, B1, B3, Q, d, Val(D), Val(D), Val(D), Val(:small))
            # B1 now contains P_pred. We keep this until the final update step.

            #####################
            #### KALMAN GAIN ####
            #####################

            # K = P_pred * H' / S where S = H * P_pred * H' + R
            # We compute K' = S^{-1} * H * P_pred, then K = K'

            # H * P_pred → B1
            batch_op!(*, B3, H, B1, d, Val(D), Val(D), Val(D), Val(:small))
            # B1 now contains H * P_pred = (P_pred * H')' since P_pred is symmetric

            # H * P_pred * H' → B2
            batch_op!(*, B2, B3, H', d, Val(D), Val(D), Val(D), Val(:small))

            # S = H*P_pred*H' + R → B2
            batch_op!(+, B2, B2, R, d, Val(D), Val(D), Val(D), Val(:small))

            # In-place Cholesky of S (B2 becomes U where S = U'*U)
            batch_op!(cholesky, B2, B2, d, Val(D), Val(D), warp_matrix_id, Val(:small))

            # In-place forward solve U' \ B1 → B1 (X = (U')^{-1} * H*P_pred)
            # We need U' for forward solve
            batch_op!(\, B3, LowerTriangular(B2'), B3, d, Val(D), Val(D), Val(D), Val(:small))
            # Backward solve U \ B1 → B1 (K' = U^{-1} * X = S^{-1} * H * P_pred)
            batch_op!(\, B3, UpperTriangular(B2), B3, d, Val(D), Val(D), Val(D), Val(:small))

            #####################
            #### UPDATE STEP ####
            #####################

            # Use P_new = (I - K*H) * P_pred form of update

            # Transpose K' in B2 to get K = P_pred * H' / S
            batch_op!(*, IAddSubSetterMatrix(B2, 1.0f0, -1.0f0), B3', H, d, Val(D), Val(D), Val(D), Val(:small))
            batch_op!(*, B3, B2, B1, d, Val(D), Val(D), Val(D), Val(:small))
            # B3 now contains P_new = (I - K*H) * P_pred
        end
    end
    
    sync_warp()
    # Write P_new (final output) - B2/shmem_2 contains P_new
    dual_to_interm_transfer!(shmem_2, B3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ps_out, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

@inline function kernel_kalman_predict!(
    Ps_out,
    Ps_in,
    A_global,
    Q_global,
    x_out,
    μ_in,
    b_in,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
    ::Val{mode},   
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_mat_elems = warp_shmem_size * n_warps
    shmem_mat_1 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
    shmem_mat_2 = CuStaticSharedArray(Float32, (shmem_mat_elems,))

    pad_interval = div(32i32, D & -D) * D
    shmem_mat_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuStaticSharedArray(Float32, (shmem_mat_size_fixed,))
    shmem_Q = CuStaticSharedArray(Float32, (shmem_mat_size_fixed,))

    shmem_vec_elems = D * n_warps * n_mats_per_warp
    shmem_vec_1 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
    shmem_vec_2 = CuStaticSharedArray(Float32, (shmem_vec_elems,))

    shmem_vec_size_fixed = D
    shmem_vec_b = CuStaticSharedArray(Float32, (shmem_vec_size_fixed,))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_global, Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_Q, Q_global, Val(D))
    end
    if wid == 3i32
        shared_vector_load!(shmem_vec_b, b_in, Val(D))
    end

    A = SharedMatrix(shmem_A, Val(D))
    Q = SharedMatrix(shmem_Q, Val(D))
    b = SharedVector(shmem_vec_b, Val(D))

    # Load P
    intermediate_layout_load!(shmem_mat_1, Ps_in, Val(D), Val(nthreads), N, Val(:small))#, Val(:lower))
    interm_to_dual_transfer!(shmem_mat_2, shmem_mat_1, Val(D), Val(nthreads), N, Val(:small))

    # Load µ
    vector_load!(shmem_vec_1, μ_in, Val(D), Val(nthreads), N)

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        B1 = DualAccessMatrix(shmem_mat_1, Val(D), warp_matrix_id, Val(:small))
        B2 = DualAccessMatrix(shmem_mat_2, Val(D), warp_matrix_id, Val(:small))
        v1 = BatchedVector(shmem_vec_1, Val(D), warp_matrix_id)
        v2 = BatchedVector(shmem_vec_2, Val(D), warp_matrix_id)

        # B1 <- A * P
        batch_op!(*, B1, A, B2, d, Val(D), Val(:small))

        # B2 <- (A * P) * A'
        batch_op!(*, B2, B1, A', d, Val(D), Val(:small))

        # B1 <- (A * P * A') + Q
        batch_op!(+, B1, B2, Q, d, Val(D), Val(:small))
        # B1 now contains P_pred
        
        # v2 <- A * µ
        batch_op!(*, v2, A, v1, d, Val(D), Val(:small))

        # v2 <- (A * µ) + b
        batch_op!(+, v2, v2, b, d, Val(D), Val(:small))
    end

    # Writing P_pred
    dual_to_interm_transfer!(shmem_mat_2, shmem_mat_1, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(
        Ps_out, shmem_mat_2, Val(D), Val(nthreads), N, Val(:small), Val(mode)#, Val(:lower),
    )

    # Writing µ_pred
    vector_write!(x_out, shmem_vec_2, Val(D), Val(nthreads), N)

    return nothing
end

@inline function kernel_kalman_update!(
    Ps_out,
    Ps_in,
    H_global,
    R_global,
    µ_out,
    x_in,
    z_in,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
    ::Val{mode},   
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_mat_elems = warp_shmem_size * n_warps
    shmem_mat_1 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
    shmem_mat_2 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
    shmem_mat_3 = CuStaticSharedArray(Float32, (shmem_mat_elems,))

    pad_interval = div(32i32, D & -D) * D
    shmem_mat_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_H = CuStaticSharedArray(Float32, (shmem_mat_size_fixed,))
    shmem_R = CuStaticSharedArray(Float32, (shmem_mat_size_fixed,))

    shmem_vec_elems = D * n_warps * n_mats_per_warp
    shmem_vec_1 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
    shmem_vec_2 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
    shmem_vec_3 = CuStaticSharedArray(Float32, (shmem_vec_elems,))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_H, H_global, Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_R, R_global, Val(D))
    end

    H = SharedMatrix(shmem_H, Val(D))
    R = SharedMatrix(shmem_R, Val(D))

    # Load P_pred
    intermediate_layout_load!(shmem_mat_1, Ps_in, Val(D), Val(nthreads), N, Val(:small))#, Val(:lower))
    interm_to_dual_transfer!(shmem_mat_3, shmem_mat_1, Val(D), Val(nthreads), N, Val(:small))

    # Load x and z
    vector_load!(shmem_vec_1, x_in, Val(D), Val(nthreads), N)
    vector_load!(shmem_vec_2, z_in, Val(D), Val(nthreads), N)

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        B1 = DualAccessMatrix(shmem_mat_1, Val(D), warp_matrix_id, Val(:small))
        B2 = DualAccessMatrix(shmem_mat_2, Val(D), warp_matrix_id, Val(:small))
        B3 = DualAccessMatrix(shmem_mat_3, Val(D), warp_matrix_id, Val(:small))
        v1 = BatchedVector(shmem_vec_1, Val(D), warp_matrix_id)
        v2 = BatchedVector(shmem_vec_2, Val(D), warp_matrix_id)
        v3 = BatchedVector(shmem_vec_3, Val(D), warp_matrix_id)

        #####################
        #### KALMAN GAIN ####
        #####################

        # K = P_pred * H' / S where S = H * P_pred * H' + R
        # We compute K' = S^{-1} * H * P_pred, then K = K'

        # H * P_pred → B1
        batch_op!(*, B1, H, B3, d, Val(D), Val(:small))
        # B1 now contains H * P_pred = (P_pred * H')' since P_pred is symmetric

        # H * P_pred * H' → B2
        batch_op!(*, B2, B1, H', d, Val(D), Val(:small))

        # S = H*P_pred*H' + R → B2
        batch_op!(+, B2, B2, R, d, Val(D), Val(:small))
        # B2 now contains S

        # In-place Cholesky of S
        batch_op!(cholesky, B2, d, Val(D), n_mats_per_warp, warp_matrix_id, Val(:small))
        # B2 now contains U where S = U' * U

        # In-place forward solve U' \ B1 → B1 (X = (U')^{-1} * H*P_pred)
        # We need U' for forward solve
        batch_op!(\, LowerTriangular(B2'), B1, d, Val(D), Val(:small))

        # Backward solve U \ B1 → B1 (K' = U^{-1} * X = S^{-1} * H * P_pred)
        batch_op!(\, UpperTriangular(B2), B1, d, Val(D), Val(:small))
        # B1 now contains K'

        #####################
        #### UPDATE STEP ####
        #####################

        # Use P_new = (I - K*H) * P_pred form of update
        # Transpose K' in B2 to get K = P_pred * H' / S
        batch_op!(*, IAddSubSetterMatrix(B2, 1.0f0, -1.0f0), B1', H, d, Val(D), Val(:small))
        # B2 now contains (I - K*H)

        # (I - K * H) * x → v3
        batch_op!(*, v3, B2, v1, d, Val(D), Val(:small))

        # K * z → v1
        batch_op!(*, v1, B1', v2, d, Val(D), Val(:small))

        # x_new = (I - K*H) * x + K * z → v1
        batch_op!(+, v1, v3, v1, d, Val(D), Val(:small))

        batch_op!(*, B1, B2, B3, d, Val(D), Val(:small))
        # B2 now contains P_new = (I - K*H) * P_pred
    end

    # Writing P_pred
    dual_to_interm_transfer!(shmem_mat_2, shmem_mat_1, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(
        Ps_out, shmem_mat_2, Val(D), Val(nthreads), N, Val(:small), Val(mode)#, Val(:lower),
    )

    # Writing µ_new
    vector_write!(µ_out, shmem_vec_1, Val(D), Val(nthreads), N)

    return nothing
end

@inline function kernel_sqrt_kalman!(
    Ss_out,
    Ss_in,
    A_glob,
    S_Q_glob,
    H_glob,
    S_R_glob,
    ::Val{D},
    ::Val{THRESH},
    ::Val{nthreads},
    n_steps::Int32,
    N::Int32,
    ::Val{:small},
) where {D,THRESH,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))

    pad_interval = div(32i32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32))
    shmem_S_Q = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + shmem_size_fixed * sizeof(Float32))
    shmem_H = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 2 * shmem_size_fixed * sizeof(Float32))
    shmem_S_R = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 3 * shmem_size_fixed * sizeof(Float32))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_glob, Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_S_Q, S_Q_glob, Val(D))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_glob, Val(D))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_S_R, S_R_glob, Val(D))
    end
    sync_threads()

    B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

    A = SharedMatrix(shmem_A, Val(D), Val(D))
    S_Q = SharedMatrix(shmem_S_Q, Val(D), Val(D))
    H = SharedMatrix(shmem_H, Val(D), Val(D))
    S_R = SharedMatrix(shmem_S_R, Val(D), Val(D))

    # Load S (lower tri)
    intermediate_layout_load!(shmem_2, Ss_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        for step in 1i32:n_steps
            ######################
            #### PREDICT STEP ####
            ######################

            # X = A * S = A * B1
            batch_op!(*, B2, A, LowerTriangular(ifelse(step == 1i32, B1, B1')), d, Val(D), Val(D), Val(D), Val(:small))

            # Form predict pre-array:
            # M_pred = [(AS)'; S_Q'] = [B2'; S_Q'] (2D x D)
            # QR of M_pred -> R, stored in B1
            M_pred = BlockMatrix_2_1(B2', S_Q', Val(D), warp_matrix_id, Val(:small))

            batch_op!(qr, B1, M_pred, d, Val(D), Val(2), Val(1), warp_matrix_id, Val(:small))
            # B1 = R = U_pred (upper tri)

            # Y = H * S_pred = H * B1'
            batch_op!(*, B2, H, LowerTriangular(B1'), d, Val(D), Val(D), Val(D), Val(:small))

            # Form update pre-array:
            # M_upd =   [S_R'   0       ]
            #           [Y'     S_pred' ]
            # =
            #           [S_R'   - ]
            #           [Y'     B1]
            M_upd = BlockMatrixLowerTrig_2_2(S_R', B2', UpperTriangular(B1), Val(D), warp_matrix_id, Val(:small))
            
            batch_op!(qr, B1, M_upd, d, Val(D), Val(THRESH), Val(2), Val(2), warp_matrix_id, Val(:small))
            # B1' = R_22' = L_new
        end
    end

    sync_warp()

    # Write S_out (final output)
    dual_to_interm_transfer!(shmem_2, LowerTriangular(B1'), Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ss_out, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

@inline function kernel_sqrt_kalman_padded!(
    Ss_out,  # all matrices are dim = D + 2
    Ss_in,
    A_glob,
    S_Q_glob,
    H_glob,
    S_R_glob,
    ::Val{D},
    ::Val{nthreads},
    n_steps::Int32,
    N::Int32,
    ::Val{:small},
) where {D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    Ddiv2 = D ÷ 2

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps

    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    
    pad_interval = div(32i32, Ddiv2 & -Ddiv2) * Ddiv2
    shmem_size_fixed = Ddiv2 * Ddiv2 + (Ddiv2 * Ddiv2 - 1i32) ÷ pad_interval

    shmem_A = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32))
    shmem_S_Q = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + shmem_size_fixed * sizeof(Float32))
    shmem_H = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 2 * shmem_size_fixed * sizeof(Float32))
    shmem_S_R = CuDynamicSharedArray(Float32, shmem_size_fixed, 2 * shmem_elems * sizeof(Float32) + 3 * shmem_size_fixed * sizeof(Float32))

    # Load fixed matrices in parallel
    if wid == 1i32
        shared_matrix_load!(shmem_A, A_glob, Val(Ddiv2))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_S_Q, S_Q_glob, Val(Ddiv2))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_glob, Val(Ddiv2))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_S_R, S_R_glob, Val(Ddiv2))
    end
    sync_threads()

    B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

    A = SharedMatrix(shmem_A, Val(Ddiv2), Val(Ddiv2))
    S_Q = SharedMatrix(shmem_S_Q, Val(Ddiv2), Val(Ddiv2))
    H = SharedMatrix(shmem_H, Val(Ddiv2), Val(Ddiv2))
    S_R = SharedMatrix(shmem_S_R, Val(Ddiv2), Val(Ddiv2))

    # Load S (lower tri) shape = (Ddiv2,Ddiv2)
    intermediate_layout_load!(shmem_2, Ss_in, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        for step in 1i32:n_steps
            ######################
            #### PREDICT STEP ####
            ######################

            # X = A * S = A * B1
            batch_op!(*, B2, A, LowerTriangular(B1), d, Val(Ddiv2), Val(Ddiv2), Val(D), Val(:small))

            # Form predict pre-array in B1:
            # M_pred = [(AS)'; S_Q'] = [B2'; S_Q'] (2D x D)
            # Each thread is responsible for row d
            @inbounds @unroll for j in 1i32:Ddiv2
                # Store (AS)' in upper half, S_Q' in lower half
                if d <= Ddiv2
                    B1[d, j] = B2[j, d]
                else
                    B1[d, j] = S_Q[j, d - Ddiv2]
                end
            end
            # B1 contains M_pred

            # QR on M_pred
            batch_op!(qr, B2, B1, d, Val(D), Val(Ddiv2), Val(D), warp_matrix_id, Val(:small))
            # B2 contains R_pred

            # Y = H * S_pred = H * B2'
            batch_op!(*, B1, H, LowerTriangular(B2'), d, Val(Ddiv2), Val(Ddiv2), Val(D), Val(:small))
            # B1 contains Y

            # Form update pre-array:
            # M_upd =   [S_R'   0       ]
            #           [Y'     S_pred' ]
            # =
            #           [S_R'   - ]
            #           [B1'     B2]
            # Each thread is responsible for row d
            S_pred = UpperTriangular(B2)
            # Order important to allow in-place movement within B2 (upper left -> lower right)
            if d > Ddiv2
                @inbounds @unroll for j in D:(-1i32):1i32
                    if j <= Ddiv2
                        B2[d, j] = B1[j, d - Ddiv2]
                    else
                        B2[d, j] = S_pred[d - Ddiv2, j - Ddiv2]
                    end
                end
            end
            if d <= Ddiv2
                @inbounds @unroll for j in D:(-1i32):1i32
                    if j <= Ddiv2
                        B2[d, j] = S_R[j, d]
                    else
                        B2[d, j] = 0.0f0
                    end
                end
            end
            # B2 contains M_upd

            # Perform QR in-place on B1
            batch_op!(qr, B2, B2, d, Val(D), Val(D), Val(D), warp_matrix_id, Val(:small))
            # B2_{bottom right} = L_new' = R_22

            # Move R_22 to upper right half for standard format
            R = LowerTriangular(B2')
            if d <= Ddiv2
                @inbounds @unroll for j in 1i32:Ddiv2
                    B1[d, j] = R[d + Ddiv2, j + Ddiv2]
                end
            end
            # B1 contains R_22' = L_new
        end
    end

    sync_warp()

    # Write S_out (final output)
    dual_to_interm_transfer!(shmem_2, B1, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ss_out, shmem_2, Val(Ddiv2), Val(Ddiv2), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end