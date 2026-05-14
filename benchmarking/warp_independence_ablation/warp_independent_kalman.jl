using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32

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
        batch_op!(*, IAddSubSetterMatrix(B2, 1.0f0, -1.0f0), B1', H, d, Val(D), Val(:small))
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

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, _, ::Val{:independent})
    D, _, N = size(P_in_cpu)
    
    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    nthreads = 2^8
    nblocks = cld(N, nthreads//32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval = div(32, D & -D) * D

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval

    shmem_bytes = sizeof(Float32) * (
        3 * shmem_elems + 4 * shmem_size_fixed
    )

    kernel = @cuda launch = false kernel_kalman!(
        P_out,
        P_in,
        A,
        Q,
        H,
        R,
        Val(Int32(D)),
        Val(Int32(nthreads)),
        Int32(N),
        Val(:small),
        Val(:indep),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)
    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $P_out,
            $P_in,
            $A,
            $Q,
            $H,
            $R,
            Val(Int32($D)),
            Val(Int32($nthreads)),
            Int32($N),
            $Val(:small),
            $Val(:indep);
            threads = $nthreads, blocks = $nblocks, shmem = $shmem_bytes,
        )
    end


    # bench_results = @benchmark begin
    #     CUDA.@sync @cuda threads = $nthreads blocks = $nblocks shmem = $shmem_bytes kernel_kalman!(
    #         $P_out,
    #         $P_in,
    #         $A,
    #         $Q,
    #         $H,
    #         $R,
    #         Val(Int32($D)),
    #         Val(Int32($nthreads)),
    #         Int32($N),
    #         $Val(:small),
    #         $Val(:indep),
    #     )
    # end

    return median(bench_results.times) / 1e9 / N
end