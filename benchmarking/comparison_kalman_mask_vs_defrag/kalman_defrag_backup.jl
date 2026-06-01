using BatchedKernels
using LinearAlgebra
using CUDA
using CUDA: i32
using Random
using BenchmarkTools

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

    (shmem_elems, d, warp_matrix_id, block_mtrx_id, grid_mtrx_id, n_mats_per_warp, n_mats_per_block) = get_shmem_elems(Val(D), Val(D), Val(D), Val(nthreads))
    (shmem_elems_xx, d_xx, warp_matrix_id_xx, block_mtrx_id_xx, grid_mtrx_id_xx, n_mats_per_warp_xx, n_mats_per_block_xx) = get_shmem_elems(Val(Dx), Val(Dx), Val(D), Val(nthreads))
    (shmem_elems_yy, d_yy, warp_matrix_id_yy, block_mtrx_id_yy, grid_mtrx_id_yy, n_mats_per_warp_yy, n_mats_per_block_yy) = get_shmem_elems(Val(Dy), Val(Dy), Val(D), Val(nthreads))

    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    shmem_3 = CuDynamicSharedArray(Float32, shmem_elems, 2 * shmem_elems * sizeof(Float32))
    shmem_4 = CuDynamicSharedArray(Float32, shmem_elems, 3 * shmem_elems * sizeof(Float32))

    shmem_F = CuDynamicSharedArray(Float32, shmem_fixed_size_xx, 4 * shmem_elems * sizeof(Float32))
    shmem_Q = CuDynamicSharedArray(Float32, shmem_fixed_size_xx, (4 * shmem_elems + shmem_fixed_size_xx) * sizeof(Float32))
    shmem_H = CuDynamicSharedArray(Float32, shmem_fixed_size_yx, (4 * shmem_elems + 2 * shmem_fixed_size_xx) * sizeof(Float32))
    shmem_R = CuDynamicSharedArray(Float32, shmem_fixed_size_yy, (4 * shmem_elems + 2 * shmem_fixed_size_xx + shmem_fixed_size_yx) * sizeof(Float32))

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

    # Load P shape: (Dx,Dx)
    warps_active = cld(n_mats_per_block, n_mats_per_warp_xx)
    if wid <= warps_active
        intermediate_layout_load!(shmem_1, P_in, Val(Dx), Val(Dx), Val(Dx), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
        interm_to_dual_transfer!(shmem_2, shmem_1, Val(Dx), Val(Dx), Val(Dx), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
    end

    sync_threads()

    if warp_matrix_id_xx <= n_mats_per_warp_xx && grid_mtrx_id_xx <= N && wid <= warps_active && block_mtrx_id_xx <= n_mats_per_block
        # Calculating which warp and how many-th matrix within the warp the current matrix belongs
        # to under the new distribution
        wid_retrieve = (block_mtrx_id_xx - 1i32) ÷ n_mats_per_warp + 1i32
        warp_matrix_id_retrieve = mod1(block_mtrx_id_xx, n_mats_per_warp)

        M1 = DualAccessMatrix(shmem_1, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        P_Dx = DualAccessMatrix(shmem_2, Val(Dx), warp_matrix_id_xx, Val(:small))
        M3 = DualAccessMatrix(shmem_3, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        P_pred = M1

        # Compute P_pred = FPF' + Q
        batch_op!(*, M3, F, P_Dx, d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))
        batch_op!(*, M1, M3, F', d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))
        batch_op!(+, P_pred, M1, Q, d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))

        # Compute H * P_pred
        HP = M3
        if Dy >= Dx
            batch_op!(*, HP, H, P_pred, d_xx, Val(Dy), Val(Dx), Val(Dx), Val(:small))
        else
            # Compute in transpose-space
            batch_op!(*, HP', P_pred', H', d_xx, Val(Dx), Val(Dx), Val(Dy), Val(:small))
        end
    end

    if Dx != Dy
        sync_threads()
    end

    # Compute S = (H P_pred H') + Q and K = P_pred H' S^{-1}
    warps_active = cld(n_mats_per_block, n_mats_per_warp_yy)
    if warp_matrix_id_yy <= n_mats_per_warp_yy && grid_mtrx_id_yy <= N && wid <= warps_active && block_mtrx_id_yy <= n_mats_per_block
        wid_retrieve = (block_mtrx_id_yy - 1i32) ÷ n_mats_per_warp + 1i32
        warp_matrix_id_retrieve = mod1(block_mtrx_id_yy, n_mats_per_warp)
        
        S = U = DualAccessMatrix(shmem_2, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        HP = DualAccessMatrix(shmem_3, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        HPH_trans = DualAccessMatrix(shmem_4, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))

        # (HP) * H'
        batch_op!(*, HPH_trans, HP, H', d_yy, Val(Dy), Val(Dx), Val(Dy), Val(:small))
        
        # S = HPH' + R
        batch_op!(+, S, HPH_trans, R, d_yy, Val(Dy), Val(Dy), Val(Dy), Val(:small))

        # Cholesky of S
        batch_op!(cholesky, U, S, d_yy, Val(Dy), Val(Dy), warp_matrix_id_yy, Val(:small))
    end

    if Dx > Dy
        sync_threads()
    end

    # Compute K = HP / S normally within (D,D)
    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        U = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        HP = K_trans = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        # Compute U' \ HP
        batch_op!(\, HP, LowerTriangular(U'), HP, d, Val(Dy), Val(Dx), Val(D), Val(:small))

        # Compute K = U \ (U' \ HP) to obtain K = HP / S
        batch_op!(\, K_trans, UpperTriangular(U), HP, d, Val(Dy), Val(Dx), Val(D), Val(:small))
    end

    if Dx < Dy
        sync_threads()
    end

    # Compute rest in (Dx,Dx)
    warps_active = cld(n_mats_per_block, n_mats_per_warp_xx)
    if warp_matrix_id_xx <= n_mats_per_warp_xx && grid_mtrx_id_xx <= N && wid <= warps_active && block_mtrx_id_xx <= n_mats_per_block
        wid_retrieve = (block_mtrx_id_xx - 1i32) ÷ n_mats_per_warp + 1i32
        warp_matrix_id_retrieve = mod1(block_mtrx_id_xx, n_mats_per_warp)
        
        P_pred = DualAccessMatrix(shmem_1, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        M2 = DualAccessMatrix(shmem_2, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        K_trans = DualAccessMatrix(shmem_3, Val(D), wid_retrieve, warp_matrix_id_retrieve, Val(:small))
        M4 = DualAccessMatrix(shmem_4, Val(Dx), warp_matrix_id_xx, Val(:small))

        # M2 <- (I - KH)
        batch_op!(*, IAddSubSetterMatrix(M2, 1.0f0, -1.0f0), K_trans', H, d_xx, Val(Dx), Val(Dy), Val(Dx), Val(:small))

        # M4 <- (I - KH) * P_pred
        batch_op!(*, M4, M2, P_pred, d_xx, Val(Dx), Val(Dx), Val(Dx), Val(:small))
    end

    sync_threads()

    if wid <= warps_active
        M = DualAccessMatrix(shmem_4, Val(Dx), warp_matrix_id_xx, Val(:small))
        dual_to_interm_transfer!(shmem_3, M, Val(Dx), Val(Dx), Val(Dx), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
        intermediate_layout_write!(P_out, shmem_3, Val(Dx), Val(Dx), Val(Dx), Val(nthreads), Val(n_mats_per_block), N, Val(:small))
    end

    return nothing
end

function kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, nthreads, ::Val{:defrag})
    Dx, _, N = size(P_out_cpu)
    Dy = size(R_cpu)[1]
    D = max(Dx, Dy)
    
    Ps_out = cu(P_out_cpu)
    Ps_in = cu(P_in_cpu)
    A = cu(A_cpu)
    Q = cu(Q_cpu)
    H = cu(H_cpu)
    R = cu(R_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval = div(32, D & -D) * D

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    # Use D to compute this instead of the true shmem_size_fixed used in the kernel for both simplicity and equality of comparison against masking
    shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval

    shmem_bytes = sizeof(Float32) * (
        4 * shmem_elems + 4 * shmem_size_fixed
    )

    kernel = @cuda launch=false kernel_kalman_defrag!(
        Ps_out, Ps_in, A, Q, H, R,
        Val(Int32(Dx)), Val(Int32(Dy)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    result_curr = @benchmark begin
        CUDA.@sync $kernel(
            $Ps_out, $Ps_in, $A, $Q, $H, $R,
            Val(Int32($Dx)), Val(Int32($Dy)), Val(Int32($D)), Val(Int32($nthreads)), Int32($N);
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(result_curr.times) / 1e9
end