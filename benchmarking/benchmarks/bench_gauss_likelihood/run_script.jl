using Magma
using Random
using JLD2

include("../../plot_benchmarks.jl")
include("../../generate_tables.jl")
include("cpu_mt.jl")
include("ours.jl")
include("gpu_mem_bound.jl")
include("jax_vmap.jl")
include("magma.jl")
include("cublas_cusolver.jl")
include("../../config/Schedule.jl")

function generate_plots(D_min::Integer, D_max::Integer, methods::Dict{Val, String}, T::Type, path::String, force::Bool)
    results =  Dict{String, Vector{Float64}}()

    cache_dir = joinpath(@__DIR__, "cache")
    isdir(cache_dir) || mkdir(cache_dir)

    Magma.LibMagma.magma_init()
    queue_ptr = Ref{Magma.LibMagma.magma_queue_t}()
    device = 0
    Magma.LibMagma.magma_queue_create_internal(
        device,
        queue_ptr,
        C_NULL,  # func
        C_NULL,  # file
        0,       # line
    )

    for (method, label) in methods
        println("Computing $label")

        curr_res = Float64[]

        for D in D_min:D_max
            method_sanitised = String(typeof(method).parameters[1])
            cache_file = joinpath(
                cache_dir,
                "gaus_likelihood_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            # println(cache_file)
            if !force && isfile(cache_file)
                @load cache_file time
                # println("  Using cached result for $label, D = $D: $time s")
            else
                Random.seed!(1234)
                nthreads = Schedule.best_nthreads("gauss_likelihood", D)                
                N = Int(ceil(1e9 / (4 * 1 * D^2)))

                x_cpu = rand(Float32, D, N)
                µ_cpu = rand(Float32, D, N)
                p_out_cpu = zeros(Float32, N)

                Σ_cpu = Array{Float32}(undef, D, D, N)
                for i in 1:N
                    Σ_i = rand(Float32, D, D) / Float32(D)
                    Σ_i = Σ_i * Σ_i' + 0.1f0 * I
                    Σ_cpu[:, :, i] = Σ_i
                end

                time = gauss_likelihood_timing(p_out_cpu, x_cpu, µ_cpu, Σ_cpu, queue_ptr, nthreads, method)

                @save cache_file time
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    Magma.LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)
    Magma.LibMagma.magma_finalize()

    plot_benchmarks(results, D_min, D_max, "Gaussian log Likelihood", "gauss_likelihood", path, ylims=(:auto, 1e-7))
    write_results_csv(results, D_min, D_max, "gauss_likelihood", path)
end

function main(force::Bool)
    methods = Dict{Val, String}(
        Val(:cpu_mt) => "CPU (multithreaded)",
        Val(:ours) => "This",
        Val(:gpu_mem_bound) => "Memory bound",
        Val(:magma) => "MAGMA",
        Val(:jax_vmap) => "JAX (vmap)",
        Val(:cublas_cusolver) => "cuBLAS/cuSOLVER",
    )

    path = "studies/benchmarks/bench_gauss_likelihood"

    generate_plots(2, 32, methods, Float32, path, force)
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)