using Test
using BenchmarkTools
using BatchedKernels
using CUDA
using CUDA: i32
using LinearAlgebra
using Plots


mem_clock = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
bus_width = attribute(device(), CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
mem_bandwidth = 2.0f0 * mem_clock * (bus_width / 8) / 1e6 * 1e9   # in B/s

# Set full shared memory usage
CUDA.cache_config!(CUDA.FUNC_CACHE_PREFER_SHARED)

# Val{:small}, Val{:large}
V = Val(:large)

# Val{:conseq}, Val{:indep}
mode = Val(:indep)

# TODO: Make the selection between small/large, conseq/indep neater, rather than with if statements
if V === Val(:small)
    D_upper_limit = 15
elseif V === Val(:large)
    D_upper_limit = 32
end

results = Vector{Tuple{Int, Float64}}()
ninety_percent_limits = Vector{Float64}()

# Throughput tests
for D in 2:32
    # Target 1GB of input data to minimise impact of L1 cache
    N_bench = Int32(ceil(1e9 / (4 * 2 * D^2)))
    if V === Val(:small)
        nthreads = 2^8
        nblocks = cld(N_bench, nthreads//32 * (32 ÷ D))
    elseif V === Val(:large)
        if mode === Val(:conseq)
            nthreads = max(2^8, 1 << (ceil(Int, log2(D^2))))
            n_mats_per_block = nthreads ÷ (D * D)
            nblocks = cld(N_bench, n_mats_per_block)    
        elseif mode === Val(:indep)
            n_cols_per_warp = max(1, prevpow(2, 32 ÷ D))
            n_elems_per_mat = D ÷ n_cols_per_warp * 32 + (D % n_cols_per_warp) * D
            n_mats_per_block = 1
            nthreads = ((n_mats_per_block * n_elems_per_mat + 31) ÷ 32) * 32
            nblocks = cld(N_bench, n_mats_per_block)
        end
    end

    CUDA.seed!(1234)
    As = CUDA.rand(Float32, D, D, N_bench)
    Bs = CUDA.rand(Float32, D, D, N_bench)
    Cs = CUDA.zeros(Float32, D, D, N_bench)

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_matmul!(
            $Cs, $As, $Bs, Val(Int32($D)), Val(Int32($nthreads)), Int32($N_bench), $V, $mode,
        )
    end

    push!(results, (D, median(bench_results).time / 1e6))

    # Verify at least 90% of theoretical speed
    bytes_total = 4 * 3 * D^2 * N_bench  # 3 matrices read/written
    achieved_bandwidth = bytes_total / (median(bench_results).time / 1e9)

    ninety_percent_limit = bytes_total / (mem_bandwidth * 0.9) * 1e3
    push!(ninety_percent_limits, ninety_percent_limit)

    debug = true
    # debug |= !((@test 0.9 < achieved_bandwidth / mem_bandwidth < 1.0) isa Test.Pass)
    if debug
        @info "Dimension: $D, result: $(achieved_bandwidth / mem_bandwidth), time: $(median(bench_results))"
    end
end

# Plotting results in bar chart
Ds = [string(r[1]) for r in results]
times = [r[2] for r in results]

V_name = String(typeof(V).parameters[1])
mode_name = String(typeof(mode).parameters[1])

gr()
bar(
    Ds,
    times;
    label = "runtime",
    title = "Runtime vs D ($V_name, $mode_name)",
    xlabel = "\$D\$",
    ylabel = "\$t\$ (ms)",
    legend = true,
    grid = true,
    size = (800, 500),
)

# Plot the 90% memory bandwidth limits
plot!(
    Ds,
    ninety_percent_limits;
    label = "90% limit",
    lw = 2,
    color = :red,
    marker = :circle,
)

filename = "runtime_vs_D_$(V_name)_$(mode_name).png"
path = joinpath(@__DIR__, "..", "benchmark_plots", filename)
savefig(path)

display(current())