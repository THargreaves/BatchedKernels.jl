using JLD2
using LinearAlgebra
using CUDA

include("../../plot_benchmarks.jl")
include("../../generate_tables.jl")
include("backsolve_orig.jl")
include("backsolve_conflict.jl")
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
                "backward_solve_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            if isfile(cache_file)
                @load cache_file time
            else
                N = Int(ceil(1e9 / (4 * 2 * D^2)))

                CUDA.seed!(1234)

                Us_cpu = zeros(T, D, D, N)
                for i in 1:N
                    U_temp = randn(T, D, D)
                    Us_cpu[:, :, i] = UpperTriangular(U_temp) + T(0.5) * I
                end
                Bs_cpu = rand(T, D, D, N)
                Cs_cpu = zeros(T, D, D, N)

                Us = cu(Us_cpu)
                Bs = cu(Bs_cpu)
                Cs = cu(Cs_cpu)

                time = backsolve_timing(Cs, Us, Bs, method)

                @save cache_file time

                CUDA.reclaim()
                GC.gc()
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    plot_benchmarks(results, D_min, D_max, "Backward Solve: Dual Access vs Naive Layout", "backsolve_bank_conflict_ablation")
    write_results_csv(results, D_min, D_max, "backsolve_bank_conflict_ablation")
end

methods = Dict{Val, String}(
    Val(:no_conflict) => "Dual Access Layout",
    Val(:conflict) => "Naive Layout",
    Val(:gpu_mem_bound) => "Memory bound",
)

generate_plots(2, 8, methods, Float32)