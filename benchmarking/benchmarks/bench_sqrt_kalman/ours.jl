using CUDA
using CUDA: i32
using BenchmarkTools
using BatchedKernels

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
        shared_matrix_load!(shmem_A, A_glob, Val(D), Val(D))
    end
    if wid == 2i32
        shared_matrix_load!(shmem_S_Q, S_Q_glob, Val(D), Val(D))
    end
    if wid == 3i32
        shared_matrix_load!(shmem_H, H_glob, Val(D), Val(D))
    end
    if wid == 4i32
        shared_matrix_load!(shmem_S_R, S_R_glob, Val(D), Val(D))
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
        ######################
        #### PREDICT STEP ####
        ######################

        # X = A * S = A * B1
        batch_op!(*, B2, A, LowerTriangular(B1), d, Val(D), Val(D), Val(D), Val(:small))

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

    sync_warp()

    # Write S_out (final output)
    dual_to_interm_transfer!(shmem_2, LowerTriangular(B1'), Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ss_out, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function sqrt_kalman_timing(Ss_out_cpu, Ss_in_cpu, A_cpu, S_Q_cpu, H_cpu, S_R_cpu, _, ::Val{THRESH}, nthreads, ::Val{:ours}) where {THRESH}
    D, _, N = size(Ss_out_cpu)
    
    Ss_out = cu(Ss_out_cpu)
    Ss_in = cu(Ss_in_cpu)
    A = cu(A_cpu)
    S_Q = cu(S_Q_cpu)
    H = cu(H_cpu)
    S_R = cu(S_R_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval = div(32, D & -D) * D

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval

    shmem_bytes = sizeof(Float32) * (
        2 * shmem_elems + 4 * shmem_size_fixed
    )

    println("compiling D=$D")
    kernel = @cuda launch=false kernel_sqrt_kalman!(
        Ss_out, Ss_in, A, S_Q, H, S_R,
        Val(Int32(D)), Val(Int32(THRESH)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    println("compiled")
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)
    println("running")
    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $Ss_out, $Ss_in, $A, $S_Q, $H, $S_R,
            Val(Int32($D)), Val(Int32($THRESH)), Val(Int32($nthreads)), Int32($N),
            $Val(:small);
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end
    println("finished D=$D")

    return median(bench_results.times) / 1e9 / N
end