using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32
using LinearAlgebra

function kernel_qr_q!(
    Qs,
    As,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
) where {D1,D2,D,nthreads}
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

    # Load A
    intermediate_layout_load!(shmem_2, As, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D1), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    R = A
    Q = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        tau = batch_op!(qr, R, A, d, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
        batch_op!(Val(:qr_Q_full), Q, R, d, tau, Val(D1), Val(D2), Val(D), warp_matrix_id, Val(:small))
    end

    sync_warp()

    # Store Q
    dual_to_interm_transfer!(shmem_1, Q, Val(min(D1, D2)), Val(D2), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Qs, shmem_1, Val(min(D1, D2)), Val(D2), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function qr_q_timing(Qs_cpu, As_cpu, _, nthreads,::Val{:ours})
    D, _, N = size(Qs_cpu)
    
    Qs = cu(Qs_cpu)
    As = cu(As_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems = warp_shmem_size * n_warps
    shmem_bytes = sizeof(Float32) * 2 * shmem_elems

    kernel = @cuda launch=false kernel_qr_q!(
        Qs, As,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $Qs, $As,
            Val(Int32($D)), Val(Int32($D)), Val(Int32($D)), Val(Int32($nthreads)), Int32($N),
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end