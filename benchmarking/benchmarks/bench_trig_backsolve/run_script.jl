using Magma
using JLD2
using Random

include("../../plot_benchmarks.jl")
include("../../generate_tables.jl")
include("cpu_mt.jl")
include("ours.jl")
include("gpu_mem_bound.jl")
include("jax_vmap.jl")
include("magma.jl")
include("cublas.jl")
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
                "backsolve_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            if !force && isfile(cache_file)
                @load cache_file time
            else
                N = Int(ceil(1e9 / (4 * 3 * D^2)))
                nthreads = Schedule.best_nthreads("trig_backsolve", D)

                Random.seed!(1234)

                Us_cpu = zeros(Float32, D, D, N)
                for i in 1:N
                    U_temp = rand(Float32, D, D)
                    Us_cpu[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I
                end
                Bs_cpu = rand(Float32, D, D, N)
                Cs_cpu = zeros(Float32, D, D, N)

                time = backsolve_timing(Cs_cpu, Us_cpu, Bs_cpu, queue_ptr, nthreads, method)

                @save cache_file time
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    Magma.LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)
    Magma.LibMagma.magma_finalize()

    plot_benchmarks(results, D_min, D_max, "Upper trig backward solve", "trig_backsolve", path)
    write_results_csv(results, D_min, D_max, "trig_backsolve", path)
end

function main(force)
    methods = Dict{Val, String}(
        Val(:cpu_mt) => "CPU (multithreaded)",
        Val(:ours) => "This",
        Val(:gpu_mem_bound) => "Memory bound",
        Val(:magma) => "MAGMA",
        Val(:jax_vmap) => "JAX (vmap)",
        Val(:cublas) => "cuBLAS",
    )

    path = "studies/benchmarks/bench_trig_backsolve"

    generate_plots(2, 32, methods, Float32, path, force)
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)