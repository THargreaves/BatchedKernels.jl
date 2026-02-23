using BatchedKernels
using BenchmarkTools
using CUDA
using CUDA: i32

@inline function kernel_forward_solve!(
    Cs, Ls, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{:small}, ::Val{mode},
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
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load L
    intermediate_layout_load!(shmem_3, Ls, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        # Create dual-access matrices
        L_mat = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

        # Perform forward solve: C = L \ B
        L = LowerTriangular(L_mat)
        batch_op!(\, C, L, B, d, Val(D), Val(:small))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N, Val(:small), Val(mode))

    return nothing
end

function forward_solve_timing(L_cpu, B_cpu, _, ::Val{:ours})
    D, _, N = size(B_cpu)
    
    L = cu(L_cpu)
    B = cu(B_cpu)

    nthreads = 2^8
    nblocks = cld(N, nthreads//32 * (32 ÷ D))

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_forward_solve!(
            $B, $L, $B, Val(Int32($D)), Val(Int32($nthreads)), Int32($N), $Val(:small), Val(:indep),
        )
    end

    return median(bench_results.times) / 1e9 / N
end