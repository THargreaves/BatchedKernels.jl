using BatchedKernels
using BenchmarkTools
using LinearAlgebra
using CUDA
using CUDA: i32

include("../memory_conflict.jl")

@inline function kernel_cholesky_conflict!(
    Us, As, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    shmem_elems = n_mats_per_block * D * D

    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))

    # Load A
    conflict_batched_layout_load!(shmem_1, As, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        A = ConflictDualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        U = ConflictDualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

        batch_op!(LinearAlgebra.cholesky, U, A, d, Val(D), Val(D), warp_matrix_id, Val(:small))
    end

    # Store result
    conflict_batched_layout_write!(Us, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function cholesky_timing(U_in_cpu, A_in_cpu, ::Val{:conflict})
    D, _, N = size(U_in_cpu)

    U = cu(U_in_cpu)
    A = cu(A_in_cpu)

    nthreads = 2^8
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    n_mats_per_block = n_warps * n_mats_per_warp

    shmem_elems = n_mats_per_block * D * D

    shmem_bytes = sizeof(Float32) * 2 * shmem_elems

    kernel = @cuda launch=false kernel_cholesky_conflict!(
        U, A, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), Val(:indep),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $U, $A, Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small), $Val(:indep);
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end