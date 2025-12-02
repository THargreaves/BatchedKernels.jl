export kernel_kalman!, kernel_kalman_predict!, kernel_kalman_update!

@inline function kernel_kalman!(
    Ps_out,
    Ps_in,
    A_global,
    Q_global,
    H_global,
    R_global,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
    ::Val{mode},
) where {D,nthreads,mode}
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
    grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    pad_interval = div(32i32, D & -D) * D
    shmem_size_fixed = D * D + (D * D - 1i32) ÷ pad_interval

    shmem_A = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_Q = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_H = CuStaticSharedArray(Float32, (shmem_size_fixed,))
    shmem_R = CuStaticSharedArray(Float32, (shmem_size_fixed,))

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
    intermediate_layout_load!(shmem_1, Ps_in, Val(D), Val(nthreads), N, Val(:small))#, Val(:lower))
    interm_to_dual_transfer!(shmem_3, shmem_1, Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        B1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        B3 = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        ######################
        #### PREDICT STEP ####
        ######################

        batch_op!(*, B2, A, B3, d, Val(D), Val(:small))
        batch_op!(*, B1, B2, A', d, Val(D), Val(:small))
        batch_op!(+, B3, B1, Q, d, Val(D), Val(:small))
        # B3 now contains P_pred. We keep this until the final update step.

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

        #####################
        #### UPDATE STEP ####
        #####################

        # Use P_new = (I - K*H) * P_pred form of update

        # Transpose K' in B2 to get K = P_pred * H' / S
        batch_op!(*, IMinusSetterMatrix(B2), B1', H, d, Val(D), Val(:small))
        batch_op!(*, B1, B2, B3, d, Val(D), Val(:small))
        # B2 now contains P_new = (I - K*H) * P_pred
    end

    # Write P_new (final output) - B2/shmem_2 contains P_new
    dual_to_interm_transfer!(shmem_2, shmem_1, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(
        Ps_out, shmem_2, Val(D), Val(nthreads), N, Val(:small), Val(mode)#, Val(:lower),
    )

    return nothing
end

function kernel_kalman_predict!(
    Ps_out,
    Ps_in,
    A_global,
    Q_global,
    μ_in,
    b_in,
    x_out,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
    ::Val{mode},   
) where {D,nthreads,mode}
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
    grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

    shmem_mat_elems = (n_mats_per_warp * D + padding) * D * n_warps
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
        batch_op!(*, v2, A, v1, d, Val(D))

        # v2 <- (A * µ) + b
        batch_op!(+, v2, v2, b, d, Val(D))
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

function kernel_kalman_update!(
    Ps_out,
    Ps_in,
    H_global,
    R_global,
    y_out,
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
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

    shmem_mat_elems = (n_mats_per_warp * D + padding) * D * n_warps
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
        batch_op!(*, IMinusSetterMatrix(B2), B1', H, d, Val(D), Val(:small))
        # B2 now contains (I - K*H)

        # (I - K * H) * x → v3
        batch_op!(*, v3, B2, v1, d, Val(D))

        # K * z → v1
        batch_op!(*, v1, B1', v2, d, Val(D))

        # x_kk = (I - K*H) * x + K * z → v1
        batch_op!(+, v1, v3, v1, d, Val(D))

        # H * x_kk → v3
        batch_op!(*, v3, H, v1, d, Val(D))

        # y = z - H*x_kk → v1
        batch_op!(-, v1, v2, v3, d, Val(D))

        batch_op!(*, B1, B2, B3, d, Val(D), Val(:small))
        # B2 now contains P_new = (I - K*H) * P_pred
    end

    # Writing P_pred
    dual_to_interm_transfer!(shmem_mat_2, shmem_mat_1, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(
        Ps_out, shmem_mat_2, Val(D), Val(nthreads), N, Val(:small), Val(mode)#, Val(:lower),
    )

    # Writing y
    vector_write!(y_out, shmem_vec_1, Val(D), Val(nthreads), N)

    return nothing
end