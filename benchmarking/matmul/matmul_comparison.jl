using Magma
using JLD2

include("../plot_benchmarks.jl")
include("../generate_tables.jl")
include("cpu_mt.jl")
include("ours.jl")
include("magma_strided.jl")
include("magma_non_strided.jl")
include("gpu_mem_bound.jl")
include("cublas_strided.jl")
include("cublas_non_strided.jl")
include("jax_vmap.jl")
include("ours_vmap.jl")

function generate_plots(D_min::Integer, D_max::Integer, methods::Dict{Val, String}, T::Type)
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
                "matmul_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            if isfile(cache_file)
                @load cache_file time
                # println("  Using cached result for $label, D = $D: $time s")
            else
                N = Int(ceil(1e9 / (4 * 2 * D^2)))

                A = rand(T, D, D, N)
                B = rand(T, D, D, N)
                C = similar(A)

                time = matmul_timing(C, A, B, queue_ptr, method)

                @save cache_file time
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    Magma.LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)
    Magma.LibMagma.magma_finalize()

    plot_benchmarks(results, D_min, D_max, "Matmul", "matmul")
    write_results_csv(results, D_min, D_max, "matmul")
end

methods = Dict{Val, String}(
    Val(:cpu_mt) => "CPU (multithreaded)",
    Val(:ours) => "Ours",
    Val(:magma_strided) => "MAGMA (strided)",
    Val(:magma_non_strided) => "MAGMA (non-strided)",
    Val(:gpu_mem_bound) => "SOL",
    Val(:cublas_strided) => "cuBLAS (strided)",
    Val(:jax_vmap) => "JAX (vmap)",
    Val(:ours_vmap) => "Ours (vmap)",
)

generate_plots(2, 15, methods, Float32)