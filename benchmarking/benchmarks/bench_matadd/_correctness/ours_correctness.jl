using CUDA
using Test
using CUDA: i32
using LinearAlgebra
using BatchedKernels

include("../ours.jl")

# Test parameters
N = 2^12 + 113
nthreads = 2^8

# Accuracy tests
for D in 2:10
    global nthreads
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    CUDA.seed!(1234)

    As = CUDA.rand(Float32, D, D, N)
    Bs = CUDA.rand(Float32, D, D, N)
    As_cpu = Array(As)
    Bs_cpu = Array(Bs)

    Cs = CUDA.zeros(Float32, D, D, N)

    nthreads = 2^8
    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
    shmem_bytes = sizeof(Float32) * 3 * shmem_elems

    kernel = @cuda launch=false kernel_matadd!(
        Cs, As, Bs,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        Cs, As, Bs,
        Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
        Val(:small),
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )

    # CPU comparison
    Cs_cpu = similar(As_cpu)
    for i in 1:N
        Cs_cpu[:, :, i] = As_cpu[:, :, i] + Bs_cpu[:, :, i]
    end
    max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
    println("D=$D, error=$max_error")
    @test max_error < 1e-5
end