using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32

include("../memory_conflict.jl")

@inline function kernel_matmul_conflict!(
    Cs,
    As,
    Bs,
    n_step::Int32,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{:small},
) where {D1,D2,D,nthreads}
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

    # Use same shared memory size as dual-access for fair comparison
    # padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)
    # shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_elems = n_mats_per_block * D * D

    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    shmem_3 = CuDynamicSharedArray(Float32, shmem_elems, 2 * shmem_elems * sizeof(Float32))

    # Load A
    conflict_batched_layout_load!(shmem_1, As, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    # Load B
    conflict_batched_layout_load!(shmem_2, Bs, Val(D2), Val(D1), Val(D), Val(nthreads), N, Val(:small))

    A = ConflictDualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    B = ConflictDualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
    C = ConflictDualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        for i in 1i32:n_step
            if isodd(i)
                batch_op!(*, C', A, B', d, Val(D1), Val(D2), Val(D), Val(:small))
            else
                batch_op!(*, A', C, B', d, Val(D1), Val(D2), Val(D), Val(:small))
            end
        end
    end

    # Write C
    if isodd(n_step)
        shmem = shmem_3
    else
        shmem = shmem_1
    end
    conflict_batched_layout_write!(Cs, shmem_3, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function matmul_timing(C_out_cpu, A_in_cpu, B_in_cpu, n_step::Int, ::Val{:conflict})
    D, _, N = size(C_out_cpu)

    C = cu(C_out_cpu)
    A = cu(A_in_cpu)
    B = cu(B_in_cpu)

    nthreads = 2^8
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    # n_mats_per_warp = 32 ÷ D
    # n_warps = nthreads ÷ 32
    # padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    # shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    # shmem_bytes = sizeof(Float32) * 3 * shmem_elems
    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    n_mats_per_block = n_warps * n_mats_per_warp

    shmem_elems = n_mats_per_block * D * D

    shmem_bytes = sizeof(Float32) * 3 * shmem_elems

    kernel = @cuda launch=false kernel_matmul_conflict!(
        C, A, B, Int32(n_step), Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $C, $A, $B, Int32($n_step), Val(Int32($D)), Val(Int32($D)), Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small);
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end