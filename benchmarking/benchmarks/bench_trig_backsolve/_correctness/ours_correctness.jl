using CUDA
using Test
using CUDA: i32
using LinearAlgebra
using BatchedKernels
using Random

include("../ours.jl")

# Test parameters
N = 2^0 + 1
nthreads = 2^8

# Accuracy tests
for D in 2:10
    Random.seed!(1234)

    Us_cpu = zeros(Float32, D, D, N)
    for i in 1:N
        U_temp = rand(Float32, D, D)
        Us_cpu[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I
    end
    Bs_cpu = rand(Float32, D, D, N)
    Us = cu(Us_cpu)
    Bs = cu(Bs_cpu)
    Cs = CUDA.zeros(Float32, D, D, N)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes = sizeof(Float32) * 3 * shmem_elems

    kernel = @cuda launch=false kernel_backsolve!(
        Cs, Us, Bs,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        Cs, Us, Bs,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
        Val(:small),
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )

    Cs_result = Array(Cs)
    Cs_cpu = similar(Cs_result)
    for i in 1:N
        Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
    end

    max_error = maximum(abs.(Cs_result .- Cs_cpu))
    println("D=$D, error=$max_error")

    @test max_error < 1e-5
end