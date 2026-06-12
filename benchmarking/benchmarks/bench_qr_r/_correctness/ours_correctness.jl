using CUDA
using Test
using CUDA: i32
using LinearAlgebra
using BatchedKernels

include("../ours.jl")

# Test parameters
N = 2^9 + 1
nthreads = 2^8

# Accuracy tests
for D in 17:25
    CUDA.seed!(1234)

    As = CUDA.rand(Float32, D, D, N)
    Rs = CUDA.zeros(Float32, D, D, N)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_bytes = sizeof(Float32) * 2 * shmem_elems

    kernel = @cuda launch=false kernel_qr_r!(
        Rs, As,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        Rs, As,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )
    Rs_res = Array(Rs)

    # CPU comparison
    Rs_cpu = similar(Rs_res)
    As_cpu = Array(As)
    for i in 1:N
        Rs_cpu[:, :, i] = UpperTriangular(qr(As_cpu[:, :, i]).R)
        Rs_res[:, :, i] = UpperTriangular(Rs_res[:, :, i])
    end
    max_error = maximum(abs.(Rs_res .- Rs_cpu))
    println("D=$D, error=$max_error")
    @test max_error < 1e-4 && !isnan(max_error)
end
