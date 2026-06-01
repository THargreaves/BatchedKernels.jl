using CUDA
using CUDA: i32
using KernelAbstractions.Extras: @unroll
using BenchmarkTools
using Dates: now

const SHMEM_WORDS = Int32(2048)
const MASK = SHMEM_WORDS - Int32(1)
const THREADS = Int32(256)
const BLOCKS = Int32(256)
const ITERS = Int32(2^14)
const UNROLL = Int32(16)


function shmem_bw_kernel!(out)
    shmem = CuStaticSharedArray(Float32, SHMEM_WORDS)
    tid = threadIdx().x
    gid = (blockIdx().x - 1i32) * THREADS + tid

    @inbounds @unroll for i in 1i32:THREADS:SHMEM_WORDS
        shmem[i] = tid
    end
    sync_threads()

    idx1 = (tid - 1i32) & MASK
    idx2 = (tid + 512i32 - 1i32) & MASK
    idx3 = (tid + 1024i32 - 1i32) & MASK
    idx4 = (tid + 1536i32 - 1i32) & MASK
    STRIDE = 32i32

    acc1 = 0.0f0
    acc2 = 0.0f0
    acc3 = 0.0f0
    acc4 = 0.0f0

    for _ in 1:ITERS
        @inbounds @unroll for _ in 1:UNROLL
            acc1 += shmem[idx1 + 1i32]
            acc2 += shmem[idx2 + 1i32]
            acc3 += shmem[idx3 + 1i32]
            acc4 += shmem[idx4 + 1i32]
            idx1 = (idx1 + STRIDE) & MASK
            idx2 = (idx2 + STRIDE) & MASK
            idx3 = (idx3 + STRIDE) & MASK
            idx4 = (idx4 + STRIDE) & MASK
        end
    end

    @inbounds out[gid + 1i32] = acc1 + acc2 + acc3 + acc4
    return nothing
end

function get_shmem_bw(blocks = Int(BLOCKS))
    out = CUDA.zeros(Float32, blocks * Int(THREADS))
    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $THREADS blocks = $blocks $shmem_bw_kernel!($out)
    end
    
    t = median(bench_results.times) / 1e9
    lds = Int(ITERS) * Int(UNROLL) * 4
    bytes = blocks * Int(THREADS) * lds * 4

    return (bw = bytes / t, time_s = t, bytes = bytes)
end

function write_shmem_bw()
    r = get_shmem_bw()
    dir = joinpath(@__DIR__)
    path = joinpath(dir, "shmem_bandwidth.csv")
    gpu  = CUDA.name(CUDA.device())
    open(path, "w") do io
        println(io, "gpu_name,shmem_bandwidth_bytes_per_s,n_blocks," *
                    "threads_per_block,iters,bytes_moved,kernel_time_s,timestamp")
        println(io, join((gpu, r.bw, Int(BLOCKS), Int(THREADS), Int(ITERS),
                          r.bytes, r.time_s, string(now())), ","))
    end
    println("wrote $path  ($(round(r.bw/1e12, digits=2)) TB/s)")
    return r.bw
end

function profile_kernel()
    out = CUDA.zeros(Float32, Int(BLOCKS) * Int(THREADS))
    CUDA.@sync @cuda threads = THREADS blocks = BLOCKS shmem_bw_kernel!(out)
end

write_shmem_bw()
# # Optimise blocks
# for blocks in (256, 512, 1024, 2048)
#     r = get_shmem_bw(blocks)
#     println("blocks=$blocks, bw=$(round(r.bw / 1e12, digits=2)) TB / s, t=$(round(r.time_s * 1e3, digits=3)) ms")
# end
# blocks=256, bw=18.29 TB / s, t=15.031 ms
# blocks=512, bw=18.29 TB / s, t=30.063 ms
# blocks=1024, bw=18.29 TB / s, t=60.123 ms
# blocks=2048, bw=18.3 TB / s, t=120.156 ms

# Profiling
# profile_kernel()
# ncu --set full --launch-count 1 --kernel-name shmem_bw_kernel_ \
#     -o hardware/shmem_bw_profile_blocks_256 \
#     julia --project=../../. hardware/measure_shmem_bw.jl
# To CSV:
# ncu -i hardware/shmem_bw_profile_blocks_256.ncu-rep --csv --page raw > hardware/shmem_bw_profile_blocks_256.csv