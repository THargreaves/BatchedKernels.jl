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
for D in 2:16
    CUDA.seed!(1234)

    As = CUDA.rand(Float32, D, D, N)
    Qs = CUDA.zeros(Float32, D, D, N)

    nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

    n_mats_per_warp = 32 ÷ D
    n_warps = nthreads ÷ 32
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_bytes = sizeof(Float32) * 2 * shmem_elems

    kernel = @cuda launch=false kernel_qr_q!(
        Qs, As,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
    )
    CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_bytes)

    CUDA.@sync kernel(
        Qs, As,
        Val(Int32(D)), Val(Int32(D)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N),
        threads=nthreads, blocks=nblocks, shmem=shmem_bytes,
    )
    Qs_res = Array(Qs)

    # CPU comparison
    Qs_cpu = similar(Qs_res)
    As_cpu = Array(As)
    for i in 1:N
        Qs_cpu[:, :, i] = Matrix(qr(As_cpu[:, :, i]).Q)
    end

    max_error = maximum(abs.(Qs_res .- Qs_cpu))
    println("D=$D, error=$max_error")
    @test max_error < 1e-4 && !isnan(max_error)
end
