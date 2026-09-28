using Magma
using JLD2
using CUDA
using BenchmarkTools
using LinearAlgebra

include("../plot_benchmarks.jl")
include("../generate_tables.jl")
include("gpu_mem_bound.jl")
include("sqrt_kalman_block.jl")
include("sqrt_kalman_pad.jl")
include("../config/Schedule.jl")

function generate_plots(D_min::Integer, D_max::Integer, methods::Dict{Val, String}, T::Type, path::String, force::Bool)
    times = Dict{Val, Vector{Float64}}()

    cache_dir = joinpath(@__DIR__, "cache")
    isdir(cache_dir) || mkdir(cache_dir)

    THRESH = 10  # For blocked 2x2 QR

    for (method, label) in methods
        println("Computing $label")

        curr_res = Float64[]

        for D in D_min:D_max
            method_sanitised = String(typeof(method).parameters[1])
            cache_file = joinpath(
                cache_dir,
                "sqrt_kalman_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            # println(cache_file)
            if !force && isfile(cache_file)
                @load cache_file time
                # println("  Using cached result for $label, D = $D: $time s")
            else
                N = Int(ceil(1e9 / (4 * 2 * D^2)))
                nthreads = Schedule.best_nthreads("sqrt_kalman", D)

                S_in = Array{Float32}(undef, D, D, N)
                for i in 1:N
                    P_i = rand(Float32, D, D) / Float32(D)
                    P_i = P_i * P_i' + 0.1f0 * I
                    S_in[:, :, i] = Float32.(Matrix(cholesky(P_i).L))
                end
                S_out = zeros(Float32, D, D, N)

                A = rand(T, D, D) / Float32(D)

                Q_elem = rand(Float32, D, D) / Float32(D)^2
                Q = Q_elem * Q_elem' + 0.01f0 * I

                H = rand(Float32, D, D) / Float32(D)
                R_elem = rand(Float32, D, D) / Float32(D)^2
                R = R_elem * R_elem' + 0.01f0 * I

                time = sqrt_kalman_timing(S_out, S_in, A, Q, H, R, 0, Val(THRESH), nthreads, method)

                @save cache_file time
            end
            push!(curr_res, time)
        end

        times[method] = curr_res
        println("Finished $label")
    end

    ratios = times[Val(:pad)] ./ times[Val(:block)]
    results = Dict{String, Vector{Float64}}("time(mask) / time(defrag)" => ratios)

    plot_benchmarks(
        results,
        D_min,
        D_max,
        "",
        "sqrt_kalman_block_vs_pad",
        path;
        ylabel = "padded time / blocked block",
        xscale = :identity,
        yscale = :identity,
        hline = 1.0,
        legend = nothing,
        xlims=(D_min, D_max),
        ylims=(0.0, 4)
    )
    write_results_csv(
        results,
        D_min,
        D_max,
        "sqrt_kalman_block_vs_pad",
        path,
    )
end

function main(force::Bool)
    methods = Dict{Val, String}(
        Val(:block) => "Blocked",
        Val(:pad) => "Padded",
    )

    path = "comparison_sqrt_kalman_block_vs_pad"

    generate_plots(2, 16, methods, Float32, path, force)
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)