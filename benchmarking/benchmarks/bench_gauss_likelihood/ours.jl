using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32

function kernel_gauss_likelihood!(
    ps_out,
    xs_in,
    µs_in,
    Σs_in,
    ::Val{D1},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
) where {D1,D,nthreads}
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
    block_matrix_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp
    grid_mtrx_id = block_matrix_id + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuDynamicSharedArray(Float32, shmem_elems)
    shmem_2 = CuDynamicSharedArray(Float32, shmem_elems, shmem_elems * sizeof(Float32))

    shmem_vec_elems = D * n_warps * n_mats_per_warp
    shmem_vec_1 = CuDynamicSharedArray(Float32, shmem_vec_elems, 2 * shmem_elems * sizeof(Float32))
    shmem_vec_2 = CuDynamicSharedArray(Float32, shmem_vec_elems, (2 * shmem_elems + shmem_vec_elems) * sizeof(Float32))

    # Load x and µ
    vector_load!(shmem_vec_1, xs_in, Val(D1), Val(D), Val(nthreads), N)
    vector_load!(shmem_vec_2, µs_in, Val(D1), Val(D), Val(nthreads), N)

    # Load Σ
    intermediate_layout_load!(shmem_2, Σs_in, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D1), Val(D1), Val(D), Val(nthreads), N, Val(:small))

    M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    v1 = BatchedVector(shmem_vec_1, Val(D), warp_matrix_id)
    v2 = BatchedVector(shmem_vec_2, Val(D), warp_matrix_id)

    log_prob = 0.0f0
    active = warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
    if active
        # δ = x - µ -> v1
        batch_op!(-, v1, v1, v2, d, Val(D1), Val(D1), Val(D), Val(:small))

        # cholesky(Σ).U -> M1
        batch_op!(cholesky, M1, M1, d, Val(D1), Val(D), warp_matrix_id, Val(:small))

        # log det of Σ -> v2
        log_det = batch_op!(Val(:log_det), UpperTriangular(M1), d, Val(D1), Val(D), warp_matrix_id, Val(:small))
        # Leader threads will have log_det in their registers, others will have garbage

        # y = L \ δ -> v1
        batch_op!(\, v1, LowerTriangular(M1'), v1, d, Val(D1), Val(D1), Val(D), warp_matrix_id, Val(:small))

        # mahal = |y|^2
        mahal_dist = batch_op!(Val(:mahal_dist), v1, d, Val(D1), Val(D), warp_matrix_id, Val(:small))
        # Leader threads will have mahal in their registers

        log_prob = -0.5f0 * (D1 * log(Float32(2π)) + log_det + mahal_dist)        
    end    

    sync_threads()

    scalar_stage!(shmem_vec_1, log_prob, lid, warp_matrix_id, block_matrix_id, active, Val(D))
    
    sync_threads()

    # v1 -> global
    scalar_write!(ps_out, shmem_vec_1, n_mats_per_block)

    return nothing
end

function gauss_likelihood_timing(ps_out_cpu, xs_in_cpu, µs_in_cpu, Σs_in_cpu, _, nthreads, ::Val{:ours})
    D, _, N = size(Σs_in_cpu)
    
    ps_out = cu(ps_out_cpu)
    xs_in = cu(xs_in_cpu)
    µs_in = cu(µs_in_cpu)
    Σs_in = cu(Σs_in_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_vec_elems = D * n_warps * n_mats_per_warp

    shmem_bytes = sizeof(Float32) * (
        2 * shmem_elems + 2 * shmem_vec_elems
    )

    kernel = @cuda launch=false kernel_gauss_likelihood!(
        ps_out, xs_in, µs_in, Σs_in,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    bench_results = @benchmark begin
        CUDA.@sync $kernel(
            $ps_out, $xs_in, $µs_in, $Σs_in,
            Val(Int32($D)), Val(Int32($D)), Val(Int32($nthreads)), Int32($N),
            threads=$nthreads, blocks=$nblocks, shmem=$shmem_bytes,
        )
    end

    return median(bench_results.times) / 1e9 / N
end