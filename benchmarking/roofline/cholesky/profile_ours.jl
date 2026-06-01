using BatchedKernels
using CUDA
using CUDA: i32
using LinearAlgebra

@inline function kernel_cholesky!(
    Us_out,
    As_in,
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

    # Load A
    intermediate_layout_load!(shmem_2, As_in, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        batch_op!(cholesky, M1, M1, d, Val(D), Val(D), warp_matrix_id, Val(:small))
    end
    
    sync_warp()

    # Write C
    dual_to_interm_transfer!(shmem_2, M1, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))
    intermediate_layout_write!(Us_out, shmem_2, Val(D), Val(D), Val(D), Val(nthreads), N, Val(:small))

    return nothing
end


function main(D::Int, n_warmups::Int, nthreads::Int)
    N = Int(ceil(1e9 / (4 * 2 * D^2)))
    T = Float32

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    As_cpu = zeros(T, D, D, N)
    for i in 1:N
        A_temp = rand(T, D, D)
        As_cpu[:, :, i] = A_temp * A_temp' + 0.1f0 * I
    end
    Us_cpu = zeros(T, D, D, N)
    As = cu(As_cpu)
    Us = cu(Us_cpu)

    shmem_elems = let
        n_mats_per_warp = 32 ÷ D
        n_warps = nthreads ÷ 32
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        warp_shmem_size * n_warps
    end
    shmem_bytes = 2 * shmem_elems * sizeof(T)

    kernel = @cuda launch=false kernel_cholesky!(
        Us, As,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    # Warm-up
    for _ in 1:n_warmups
        CUDA.@sync kernel(
            Us, As,
            Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
            threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
        )
    end

    # Profile
    CUDA.@sync kernel(
        Us, As,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small);
        threads = nthreads, blocks = nblocks, shmem = shmem_bytes,
    )
    CUDA.synchronize()
end


D = parse(Int, ARGS[1])
n_warmups = parse(Int, ARGS[2])
nthreads = parse(Int, ARGS[3])
main(D, n_warmups, nthreads)