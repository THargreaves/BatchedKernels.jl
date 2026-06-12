using JLD2
using LinearAlgebra
using CUDA
using Random

include("../../plot_benchmarks.jl")
include("../../generate_tables.jl")
include("repeated_mul_defrag.jl")
include("repeated_mul_mask.jl")
include("../config/Schedule.jl")

function generate_plots(
    D1::Integer,
    D::Integer,
    max_n_muls::Integer,
    methods::Dict{Val, String},
    T::Type,
    path::String,
    force::Bool
)
    times = Dict{Val, Vector{Float64}}()

    cache_dir = joinpath(@__DIR__, "cache_D1_$(D1)_D_$(D)")
    isdir(cache_dir) || mkdir(cache_dir)

    Random.seed!(1234)
    nthreads = Schedule.best_nthreads("matmul", D)
    N = Int(ceil(1e9 / (4 * 2 * D1^2)))

    M_in_cpu = rand(T, D1, D1, N)
    A_cpu = rand(T, D1, D1)
    M_out_cpu = zeros(T, D1, D1, N)

    for (method, label) in methods
        println("Computing $label")

        curr_res = Float64[]

        for n_muls in 1:max_n_muls
            method_sanitised = String(typeof(method).parameters[1])
            cache_file = joinpath(
                cache_dir,
                "repeated_mul_$(string(T))_$(method_sanitised)_n_muls_$(n_muls).jld2",
            )
            if !force && isfile(cache_file)
                @load cache_file time
            else
                time = matmul_timing(
                    M_out_cpu, M_in_cpu, A_cpu, n_muls, Val(D), nthreads, method,
                )

                @save cache_file time

                CUDA.reclaim()
                GC.gc()
            end
            push!(curr_res, time)
        end

        times[method] = curr_res
        println("Finished $label")
    end

    ratios = times[Val(:mask)] ./ times[Val(:defrag)]
    results = Dict{String, Vector{Float64}}("time(mask) / time(defrag)" => ratios)

    plot_benchmarks(
        results,
        1,
        max_n_muls,
        "Masking vs Defragmentation (D1=$D1, D=$D)",
        "comparison_repeated_mul_mask_vs_defrag_$(D1)_D_$(D)",
        path;
        xlabel = "n_muls",
        ylabel = "time(mask) / time(defrag)",
        xscale = :identity,
        yscale = :identity,
        hline = 2.0,
        legend = :bottomright
    )
    write_results_csv(
        results,
        1,
        max_n_muls,
        "comparison_repeated_mul_mask_vs_defrag_D1_$(D1)_D_$(D)",
        path,
    )
end

function main(force)
    methods = Dict{Val, String}(
        Val(:mask) => "Masking",
        Val(:defrag) => "Defragmentation",
    )

    path = "studies/comparison_repeated_mul_mask_vs_defrag"

    generate_plots(4, 8, 32, methods, Float32, path, force)
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)