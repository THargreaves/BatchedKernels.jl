using BatchedKernels
using CUDA
using CUDA: i32
using BenchmarkTools
using KernelAbstractions.Extras: @unroll

@inline function opaque_barrier()
    Base.llvmcall(
        """
        call void asm sideeffect "", "~{memory}"()
        ret void
        """,
        Cvoid, Tuple{}
    )
end

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

@inline function kernel_mul_mask!(
    M_out,
    M_in,
    A_global,
    ::Val{Dx},
    ::Val{Dy},
    ::Val{D},
    ::Val{nthreads},
    ::Val{n_muls},
    N::Int32,
) where {Dx,Dy,D,nthreads,n_muls}
    tid = threadIdx().x
    wid = div(tid - 1i32, 32i32) + 1i32

    pad_interval_yx = div(32i32, Dy & -Dy) * Dy
    shmem_fixed_size_yx = Dy * Dx + (Dy * Dx - 1i32) ÷ pad_interval_yx

    shmem_A = CuStaticSharedArray(Float32, (shmem_fixed_size_yx,))

    if wid == 1i32
        shared_matrix_load!(shmem_A, A_global, Val(Dy), Val(Dx))
    end

    A = SharedMatrix(shmem_A, Val(Dy), Val(Dx))

    (shmem_elems, d, warp_matrix_id, block_mtrx_id, grid_mtrx_id, n_mats_per_warp, n_mats_per_block) = get_shmem_elems(Val(D), Val(D), Val(D), Val(nthreads))

    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))

    M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
    M2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

    # Load M_in
    intermediate_layout_load!(shmem_2, M_in, Val(Dx), Val(Dy), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(Dx), Val(Dy), Val(D), Val(nthreads), N, Val(:small))

    sync_threads()

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        @unroll for i in 1i32:n_muls
            batch_op!(*, M2, M1, A, d, Val(Dx), Val(Dy), Val(Dx), Val(:small))
            opaque_barrier()
        end
    end

    sync_warp()
    
    M = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
    dual_to_interm_transfer!(shmem_1, M, Val(Dx), Val(Dx), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(M_out, shmem_1, Val(Dx), Val(Dx), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end

function matmul_timing(M_out_cpu, M_in_cpu, A_cpu, n_muls, nthreads, ::Val{:mask})
    Dx, Dy, N = size(M_in_cpu)
    D = max(Dx, Dy)
    
    Ms_out = cu(M_out_cpu)
    Ms_in = cu(M_in_cpu)
    A = cu(A_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks $kernel_mul_mask!(
            $Ms_out, $Ms_in, $A,
            Val(Int32($Dx)), Val(Int32($Dy)), Val(Int32($D)), Val(Int32($nthreads)),
            Val(Int32($n_muls)), Int32($N),
        )
    end

    return median(bench_results.times) / 1e9 / N
end