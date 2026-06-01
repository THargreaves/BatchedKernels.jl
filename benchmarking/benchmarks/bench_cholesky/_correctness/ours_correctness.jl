using CUDA
using Test
using CUDA: i32
using LinearAlgebra
using BatchedKernels
using Random

include("../ours.jl")

# Test parameters
N = 2^12 + 113
nthreads = 2^8

# Accuracy tests
for D in 2:10
    Random.seed!(1234)

    As_cpu = zeros(Float32, D, D, N)
    for i in 1:N
        A_temp = rand(Float32, D, D)
        As_cpu[:, :, i] = A_temp * A_temp' + 0.1f0 * I
    end
    Us_cpu = zeros(Float32, D, D, N)

    As = cu(As_cpu)
    Us = cu(Us_cpu)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes = sizeof(Float32) * 2 * shmem_elems

    kernel = @cuda launch=false kernel_cholesky!(
        Us, As,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        Us, As,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
        Val(:small),
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )

    # CPU comparison
    Us_result = Array(Us)
    for i in 1:N
        Us_cpu[:, :, i] = Matrix(cholesky(As_cpu[:, :, i]).U)
        Us_result[:, :, i] = UpperTriangular(Us_result[:, :, i])
    end
    max_error = maximum(abs.(Us_result .- Us_cpu))
    println("D=$D, error=$max_error")
    @test max_error < 1e-5
end