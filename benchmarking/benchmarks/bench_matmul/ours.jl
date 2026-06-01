using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32

@inline function kernel_matmul!(
    Cs_out,
    As_in,
    Bs_in,
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

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))
    shmem_3 = CuDynamicSharedArray(Float32, shmem_elems, 2 * shmem_elems * sizeof(Float32))

    # Load A
    intermediate_layout_load!(shmem_3, As_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    M2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
    M3 = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        batch_op!(*, M3, M1, M2, d, Val(D), Val(D), Val(D), Val(:small))
    end
    
    sync_warp()

    # Write C
    dual_to_interm_transfer!(shmem_1, M3, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs_out, shmem_1, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function matmul_timing(C_out_cpu, A_in_cpu, B_in_cpu, _, nthreads, ::Val{:ours})
    D, _, N = size(C_out_cpu)
    
    C_out = cu(C_out_cpu)
    A_in = cu(A_in_cpu)
    B_in = cu(B_in_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_bytes = sizeof(Float32) * 3 * shmem_elems

    if D > 16
        kernel = @cuda launch=false maxregs=96 kernel_matmul!(
            C_out, A_in, B_in,
            Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
        )
    else
        kernel = @cuda launch=false kernel_matmul!(
            C_out, A_in, B_in,
            Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
        )
    end
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $C_out, $A_in, $B_in,
            Val(Int32($D)), Val(Int32($nthreads)), Int32($N),
            $Val(:small),
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end