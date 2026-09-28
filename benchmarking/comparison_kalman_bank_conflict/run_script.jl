using JLD2
using LinearAlgebra
using Random

include("../plot_benchmarks.jl")
include("../generate_tables.jl")
include("conflcit_kalman.jl")
include("original_kalman.jl")
include("../config/Schedule.jl")

function generate_plots(D_min::Integer, D_max::Integer, n_steps::Int, path::String, force::Bool)
    cache_dir = joinpath(@__DIR__, "cache_$(n_steps)")
    isdir(cache_dir) || mkdir(cache_dir)

    results =  Dict{String, Vector{Float64}}()
    curr_res = Float64[]

    for D in D_min:D_max
        cache_file = joinpath(
            cache_dir,
            "kalman_D_$(D).jld2",
        )
        if !force && isfile(cache_file)
            @load cache_file ratio
        else
            Random.seed!(1234)
            T = Float32
            N = Int(ceil(1e9 / (4 * 2 * D^2)))
            nthreads = 128

            A_cpu = rand(T, D, D, N) / T(D)
            Q_cpu = zeros(T, D, D, N)
            for i in 1:N
                Q_elem = rand(T, D, D) / T(D)^2
                Q_cpu[:, :, i] = Q_elem * Q_elem' + 0.01f0 * I
            end

            H_cpu = rand(T, D, D, N) / T(D)
            R_cpu = zeros(T, D, D, N)
            for i in 1:N
                R_elem = rand(T, D, D) / T(D)^2
                R_cpu[:, :, i] = R_elem * R_elem' + 0.01f0 * I
            end

            P_in_cpu = Array{T}(undef, D, D, N)
            for i in 1:N
                P_i = rand(T, D, D) / T(D)
                P_i = P_i * P_i' + 0.1f0 * I
                P_in_cpu[:, :, i] = P_i
            end

            P_out_cpu = zeros(T, D, D, N)

            median_orig = kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, nthreads, Val(:orig))
            median_conflict = kalman_timing(P_out_cpu, P_in_cpu, A_cpu, Q_cpu, H_cpu, R_cpu, nthreads, Val(:conflict))

            ratio = median_orig / median_conflict

            @save cache_file ratio

            CUDA.reclaim()
            GC.gc()
        end
        push!(curr_res, ratio)
    end

    results["ratio"] = curr_res

    suffix = n_steps == 1. ? "" : "s"

    plot_benchmarks(
        results,
        D_min,
        D_max,
        "",
        "comparison_kalman_bank_conflict_n_steps_$n_steps",
        path;
        ylabel = "dual access time / naive layout time",
        xscale = :identity,
        yscale = :identity,
        legend = false,
        ylims=(0.0, 1.5),
    )
    write_results_csv(results, D_min, D_max, "comparison_kalman_bank_conflict_n_steps_$n_steps", path)
end

function main(force::Bool)
    path = "comparison_kalman_bank_conflict"

    # Not configured for multi-step!
    for n_steps in [1,]
        generate_plots(2, 32, n_steps, path, force)
    end
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)