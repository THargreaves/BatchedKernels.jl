using Test
using BenchmarkTools
using BatchedKernels
using CUDA
using CUDA: i32
using LinearAlgebra
using Plots

include("kalman_kernels.jl")

mem_clock = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
bus_width = attribute(device(), CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
mem_bandwidth = 2.0f0 * mem_clock * (bus_width / 8) / 1e6 * 1e9   # in B/s

# Set full shared memory usage
CUDA.cache_config!(CUDA.FUNC_CACHE_PREFER_SHARED)

# Val{:conseq}, Val{:indep}
mode = Val(:indep)

results = Vector{Tuple{Int,Float64}}()
ninety_percent_limits = Vector{Float64}()

# Throughput tests
for D in 2:13
    # Target 1GB of input data to minimise impact of L1 cache
    N_bench = Int32(ceil(1e9 / (4 * 2 * D^2)))
    nthreads = 2^8
    nblocks = cld(N_bench, nthreads//32 * (32 ÷ D))

    CUDA.seed!(1234)
   
    # Generate test data
    A_elem = rand(Float32, D, D) / Float32(D)
    Q_elem = rand(Float32, D, D) / Float32(D)^2
    Q_elem = Q_elem * Q_elem' + 0.01f0 * I

    H_elem = rand(Float32, D, D) / Float32(D)
    R_elem = rand(Float32, D, D) / Float32(D)^2
    R_elem = R_elem * R_elem' + 0.01f0 * I

    P_cpu = Array{Float32}(undef, D, D, N_bench)
    for i in 1:N_bench
        P_i = rand(Float32, D, D) / Float32(D)
        P_i = P_i * P_i' + 0.1f0 * I
        P_cpu[:, :, i] = P_i
    end

    P_in = CuArray(P_cpu)

    A_gpu = CuArray(A_elem)
    Q_gpu = CuArray(Q_elem)
    H_gpu = CuArray(H_elem)
    R_gpu = CuArray(R_elem)

    µ_cpu = rand(Float32, D, N_bench)
    b_cpu = rand(Float32, D)
    z_cpu = rand(Float32, D, N_bench)

    µ_gpu = cu(µ_cpu)
    b_gpu = cu(b_cpu)
    z_gpu = cu(z_cpu)

    P_out = CuArray{Float32}(undef, D, D, N_bench)
    µ_out = CuArray{Float32}(undef, D, N_bench)

    bench_results = @benchmark begin
        CUDA.@sync @cuda threads = $nthreads blocks = $nblocks kernel_kalman!(
            $P_out,
            $P_in,
            $A_gpu,
            $Q_gpu,
            $H_gpu,
            $R_gpu,
            $µ_out,
            $µ_gpu,
            $b_gpu,
            $z_gpu,
            Val(Int32($D)),
            Val(Int32($nthreads)),
            Int32($N_bench),
            $
            $mode,
        )
    end

    push!(results, (D, median(bench_results).time / 1e6))

    # Verify at least 90% of theoretical speed
    bytes_total = 4 * 2 * D^2 * N_bench  # 2 matrices (P_in, P_out) read/written
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

mode_name = String(typeof(mode).parameters[1])

gr()
bar(
    Ds,
    times;
    label="runtime",
    title="Runtime vs D (small, $mode_name)",
    xlabel="\$D\$",
    ylabel="\$t\$ (ms)",
    legend=true,
    grid=true,
    size=(800, 500),
)

# Plot the 90% memory bandwidth limits
plot!(Ds, ninety_percent_limits; label="90% limit", lw=2, color=:red, marker=:circle)

filename = "kalman_small_$(mode_name).png"
path = joinpath(@__DIR__, "..", "..", "throughput_plots", filename)
savefig(path)

display(current())
