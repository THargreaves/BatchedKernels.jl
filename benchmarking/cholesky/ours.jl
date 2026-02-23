using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32

@inline function kernel_cholesky_inplace!(
    Us, As, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
) where {D,nthreads,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load A
    intermediate_layout_load!(shmem_2, As, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))

        # Perform out-of-place Cholesky
        batch_op!(cholesky, A, d, Val(D), n_mats_per_warp, warp_matrix_id, Val(:small))
    end

    # Store result
    dual_to_interm_transfer!(shmem_2, shmem_1, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Us, shmem_2, Val(D), Val(nthreads), N, Val(:small), Val(mode))

    return nothing
end

function cholesky_timing(A_cpu, _, ::Val{:ours})
    D, _, N = size(A_cpu)
    T = eltype(A_cpu)
    
    A = cu(A_cpu)
    U = CUDA.zeros(T, D, D, N)

    nthreads = 2^8
    nblocks = cld(N, nthreads//32 * (32 ÷ D))

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_cholesky_inplace!(
            $U, $A, Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small), Val(:indep),
        )
    end

    return median(bench_results.times) / 1e9 / N
end