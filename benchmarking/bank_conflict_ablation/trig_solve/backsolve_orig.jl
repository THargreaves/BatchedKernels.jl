using BatchedKernels
using CUDA
using CUDA: i32

@inline function kernel_backward_solve_orig!(
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
    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    shmem_3 = CuDynamicSharedArray(Float32, shmem_elems, 2 * shmem_elems * sizeof(Float32))

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
        batch_op!(\, C, U, B, d, Val(D1), Val(D2), Val(D), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function backsolve_timing(C_out_cpu, U_in_cpu, B_in_cpu, ::Val{:no_conflict})
    D, _, N = size(C_out_cpu)
    
    Cs = cu(C_out_cpu)
    Us = cu(U_in_cpu)
    Bs = cu(B_in_cpu)

    nthreads = 2^8
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps

    shmem_bytes = sizeof(Float32) * 3 * shmem_elems

    kernel = @cuda launch=false kernel_backward_solve_orig!(
        Cs, Us, Bs, Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $Cs, $Us, $Bs, Val(Int32($D)), Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small),
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end