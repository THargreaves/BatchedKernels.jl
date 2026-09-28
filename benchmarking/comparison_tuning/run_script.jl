using Magma
using JLD2
using LinearAlgebra
using Random

include("../plot_benchmarks.jl")
include("../generate_tables.jl")
include("kalman_untuned.jl")
include("kalman_tuned.jl")
include("gpu_mem_bound.jl")
include("../config/Schedule.jl")

function generate_plots(D_min::Integer, D_max::Integer, n_steps::Int, T::Type, path::String, force::Bool)
    results =  Dict{String, Vector{Float64}}()

    cache_dir = joinpath(@__DIR__, "cache_$(n_steps)")
    isdir(cache_dir) || mkdir(cache_dir)

    curr_res = Float64[]

    for D in D_min:D_max
        cache_file = joinpath(
            cache_dir,
            "comparison_tuning_D_$(D).jld2",
        )
        if !force && isfile(cache_file)
            @load cache_file ratio
        else
            Random.seed!(1234)
            N = Int(ceil(1e9 / (4 * 2 * D^2)))
            nthreads = Schedule.best_nthreads("kalman", D)

            P_out = zeros(T, D, D, N)
            P_in = zeros(T, D, D, N)
            for i in 1:N
                P_i = rand(T, D, D) / T(D)
                P_i = P_i * P_i' + 0.1f0 * I
                P_in[:, :, i] = P_i
            end

            A = T.(Matrix(qr(randn(T, D, D)).Q))

            Q_elem = rand(Float32, D, D) / Float32(D)^2
            Q = Q_elem * Q_elem' + 0.01f0 * I

            H = rand(Float32, D, D) / Float32(D)
            R_elem = rand(Float32, D, D) / Float32(D)^2
            R = R_elem * R_elem' + 0.01f0 * I

            tuned = kalman_timing(P_out, P_in, A, Q, H, R, n_steps, nthreads, Val(:tuned))
            untuned = kalman_timing(P_out, P_in, A, Q, H, R, n_steps, nthreads, Val(:untuned))

            ratio = tuned / untuned

            @save cache_file ratio

            CUDA.reclaim()
            GC.gc()
        end
        push!(curr_res, ratio)
    end

    results["ratio"] = curr_res

    plot_benchmarks(
        results,
        D_min,
        D_max,
        "",
        "comparison_tuning",
        path;
        ylabel = "tuned time / untuned time",
        hline = 1.0,
        xscale = :identity,
        yscale = :identity,
        legend = false,
        ylims=(0.0, 1.1),
    )
    write_results_csv(results, D_min, D_max, "comparison_tuning", path)
end

function main(force::Bool)
    path = "comparison_tuning"

    generate_plots(2, 32, 1, Float32, path, force)
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)