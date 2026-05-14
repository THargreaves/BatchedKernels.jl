using JLD2
using LinearAlgebra
using CUDA
using Random

include("../../plot_benchmarks.jl")
include("../../generate_tables.jl")
include("original_cholesky.jl")
include("conflict_cholesky.jl")
include("gpu_mem_bound.jl")

function generate_plots(D_min::Integer, D_max::Integer, methods::Dict{Val, String}, T::Type)
    results = Dict{String, Vector{Float64}}()

    cache_dir = joinpath(@__DIR__, "cache")
    isdir(cache_dir) || mkdir(cache_dir)

    for (method, label) in methods
        println("Computing $label")

        curr_res = Float64[]

        for D in D_min:D_max
            method_sanitised = String(typeof(method).parameters[1])
            cache_file = joinpath(
                cache_dir,
                "cholesky_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            if isfile(cache_file)
                @load cache_file time
            else
                N = Int(ceil(1e9 / (4 * 2 * D^2)))

                Random.seed!(1234)

                # Create symmetric positive definite matrices
                As_cpu = zeros(Float32, D, D, N)
                for i in 1:N
                    A_temp = rand(Float32, D, D)
                    As_cpu[:, :, i] = A_temp * A_temp' + 0.1f0 * I
                end
                Us_cpu = zeros(Float32, D, D, N)

                As = cu(As_cpu)
                Us = cu(Us_cpu)

                time = cholesky_timing(Us, As, method)

                @save cache_file time

                CUDA.reclaim()
                GC.gc()
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    plot_benchmarks(results, D_min, D_max, "Cholesky: Dual Access vs Naive Layout", "cholesky_bank_conflict_ablation")
    write_results_csv(results, D_min, D_max, "cholesky_bank_conflict_ablation")
end

methods = Dict{Val, String}(
    Val(:no_conflict) => "Dual Access Layout",
    Val(:conflict) => "Naive Layout",
    Val(:gpu_mem_bound) => "Memory bound",
)

generate_plots(2, 8, methods, Float32)