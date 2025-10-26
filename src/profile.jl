using CUDA
using CUDA: i32
using BatchedKernels
using BenchmarkTools

CUDA.cache_config!(CUDA.FUNC_CACHE_PREFER_SHARED)

D = 31
N = Int32(ceil(1e9 / (4 * 2 * D^2)))

if BatchedKernels.VERSION === :NMatsPerWarp
    nthreads = 2^8
    nblocks = cld(N, nthreads//32 * (32 ÷ D))
elseif BatchedKernels.VERSION === :D2ThreadsPerMat
    nthreads = 1 << (ceil(Int, log2(D^2)))
    n_mats_per_block = nthreads ÷ (D * D)
    nblocks = cld(N, n_mats_per_block)  
elseif BatchedKernels.VERSION === :OneMatPerWarp
    nthreads = 2^8
    n_mats_per_warp = 1
    n_warps = nthreads ÷ 32
    n_mats_per_block = n_warps * n_mats_per_warp
    nblocks = cld(N, n_mats_per_block)
end

CUDA.seed!(1234)
As = CUDA.rand(Float32, D, D, N)
Bs = CUDA.rand(Float32, D, D, N)

Cs = CUDA.zeros(Float32, D, D, N)

CUDA.@profile begin
    CUDA.@sync @cuda threads=nthreads blocks=nblocks kernel_matmul!(
        Cs, As, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
    )
end


# Print SASS
# @device_code_sass @cuda threads=nthreads blocks=nblocks kernel_matmul!(
#     Cs, As, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
# )


# ncu \
#   --set full \
#   --profile-from-start off \
#   --target-processes all \
#   --import-source yes \
#   --kernel-name-base demangled \
#   -o ncu_report -f \
#   julia --project=. src/profile.jl