using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32
using LinearAlgebra

@inline function kernel_kalman_orig!(
    Ps_out,
    Ps_in,
    As_in,
    Qs_in,
    Hs_in,
    Rs_in,
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
) where {D,nthreads}
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
    active = warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_M1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_M2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    shmem_M3 = CuDynamicSharedArray(Float32, shmem_elems, 2 * shmem_elems * sizeof(Float32))
    shmem_M4 = CuDynamicSharedArray(Float32, shmem_elems, 3 * shmem_elems * sizeof(Float32))
    shmem_M5 = CuDynamicSharedArray(Float32, shmem_elems, 4 * shmem_elems * sizeof(Float32))

    M1 = DualAccessMatrix(shmem_M1, Val(D), warp_matrix_id, Val(:small))
    M2 = DualAccessMatrix(shmem_M2, Val(D), warp_matrix_id, Val(:small))
    M3 = DualAccessMatrix(shmem_M3, Val(D), warp_matrix_id, Val(:small))
    M4 = DualAccessMatrix(shmem_M4, Val(D), warp_matrix_id, Val(:small))
    M5 = DualAccessMatrix(shmem_M5, Val(D), warp_matrix_id, Val(:small))

    # Load A
    intermediate_layout_load!(shmem_M2, As_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_M1, shmem_M2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    # Load P_in
    intermediate_layout_load!(shmem_M3, Ps_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_M2, shmem_M3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    
    if active
        # A * P * A'
        batch_op!(*, M3, M1, M2, d, Val(D), Val(D), Val(D), Val(:small))
        batch_op!(*, M2, M3, M1', d, Val(D), Val(D), Val(D), Val(:small))
    end

    # Load Q
    intermediate_layout_load!(shmem_M4, Qs_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_M3, shmem_M4, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    if active
        # (A * P * A' + Q) -> P_pred
        batch_op!(+, M2, M2, M3, d, Val(D), Val(D), Val(D), Val(:small))
    end

    # Load H
    intermediate_layout_load!(shmem_M3, Hs_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_M1, shmem_M3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    if active
        # H * P_pred * H' -> M4
        batch_op!(*, M3, M1, M2, d, Val(D), Val(D), Val(D), Val(:small))
        batch_op!(*, M4, M3, M1', d, Val(D), Val(D), Val(D), Val(:small))
    end

    # Load R
    intermediate_layout_load!(shmem_M5, Rs_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_M3, shmem_M5, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    if active
        # (H * P * H') + R -> M4
        batch_op!(+, M4, M4, M3, d, Val(D), Val(D), Val(D), Val(:small))

        # H * P_pred (again) -> M3
        batch_op!(*, M3, M2, M1', d, Val(D), Val(D), Val(D), Val(:small))

        # Cholesky S -> M4
        batch_op!(cholesky, M4, Symmetric(M4), d, Val(D), Val(D), warp_matrix_id, Val(:small))

        # Two triangular solves
        batch_op!(\, M3, LowerTriangular(M4'), M3', d, Val(D), Val(D), Val(D), Val(:small))
        batch_op!(\, M3, UpperTriangular(M4), M3, d, Val(D), Val(D), Val(D), Val(:small))

        # K * H -> M4
        batch_op!(*, M4, M3', M1, d, Val(D), Val(D), Val(D), Val(:small))

        # (I - KH) * P_pred
        batch_op!(*, M3, IAddSubGetterMatrix(M4, 1.0f0, -1.0f0), M2, d, Val(D), Val(D), Val(D), Val(:small))
    end
    
    # Store P_out
    dual_to_interm_transfer!(shmem_M5, M3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Ps_out, shmem_M5, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
end

function kalman_timing(P_out_cpu, P_in_cpu, As_cpu, Qs_cpu, Hs_cpu, Rs_cpu, nthreads, ::Val{:orig})
    D, _, N = size(P_in_cpu)
    
    P_out = cu(P_out_cpu)
    P_in = cu(P_in_cpu)
    As = cu(As_cpu)
    Qs = cu(Qs_cpu)
    Hs = cu(Hs_cpu)
    Rs = cu(Rs_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps

    shmem_bytes = sizeof(Float32) * 5 * shmem_elems

    kernel = @cuda launch=false kernel_kalman_orig!(
        P_out, P_in, As, Qs, Hs, Rs,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $P_out, $P_in, $As, $Qs, $Hs, $Rs,
            Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small),
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end