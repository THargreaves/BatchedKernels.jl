using Test
using BenchmarkTools
using BatchedKernels
using CUDA
using CUDA: i32
using LinearAlgebra


mem_clock = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
bus_width = attribute(device(), CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
mem_bandwidth = 2.0f0 * mem_clock * (bus_width / 8) / 1e6 * 1e9   # in B/s

# Set full shared memory usage
CUDA.cache_config!(CUDA.FUNC_CACHE_PREFER_SHARED)

if BatchedKernels.VERSION === :NMatsPerWarp
    D_upper_limit = 15
elseif BatchedKernels.VERSION === :D2ThreadsPerMat
    D_upper_limit = 32
elseif BatchedKernels.VERSION === :OneMatPerWarp
    D_upper_limit = 15
end

# Throughput tests
for D in 2:D_upper_limit
    # Target 1GB of input data to minimise impact of L1 cache
    N_bench = Int32(ceil(1e9 / (4 * 2 * D^2)))
    if BatchedKernels.VERSION === :NMatsPerWarp
        nthreads = 2^8
        nblocks = cld(N_bench, nthreads//32 * (32 ÷ D))
    elseif BatchedKernels.VERSION === :D2ThreadsPerMat
        nthreads = max(2^8, 1 << (ceil(Int, log2(D^2))))
        n_mats_per_block = nthreads ÷ (D * D)
        nblocks = cld(N_bench, n_mats_per_block)    
    elseif BatchedKernels.VERSION === :OneMatPerWarp
        nthreads = 2^8
        n_mats_per_warp = 1
        n_warps = nthreads ÷ 32
        n_mats_per_block = n_warps * n_mats_per_warp
        nblocks = cld(N_bench, n_mats_per_block)
    end

    CUDA.seed!(1234)
    As = CUDA.rand(Float32, D, D, N_bench)
    Bs = CUDA.rand(Float32, D, D, N_bench)
    Cs = CUDA.zeros(Float32, D, D, N_bench)

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_matmul!(
            $Cs, $As, $Bs, Val(Int32($D)), Val(Int32($nthreads)), Int32($N_bench)
        )
    end

    # Verify at least 90% of theoretical speed
    bytes_total = 4 * 3 * D^2 * N_bench  # 3 matrices read/written
    achieved_bandwidth = bytes_total / (median(bench_results).time / 1e9)
    debug = true
    # debug |= !((@test 0.9 < achieved_bandwidth / mem_bandwidth < 1.0) isa Test.Pass)
    if debug
        @info "Dimension: $D, result: $(achieved_bandwidth / mem_bandwidth), time: $(median(bench_results))"
    end
end