using Magma
using LinearAlgebra
using JLD2

include("../plot_benchmarks.jl")
include("cpu_mt.jl")
include("gpu_mem_bound.jl")
include("ours.jl")
include("magma_non_strided.jl")
include("cublas_non_strided.jl")

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
                "solve_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            if isfile(cache_file)
                @load cache_file time
            else
                N = Int(ceil(1e9 / (4 * 2 * D^2)))

                L = zeros(T, D, D, N)
                for i in 1:N
                    L_temp = rand(T, D, D)
                    L[:, :, i] = LowerTriangular(L_temp) + 0.5f0 * I
                end
                B = rand(T, D, D, N)

                time = forward_solve_timing(L, B, queue_ptr, method)

                @save cache_file time
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    Magma.LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)
    Magma.LibMagma.magma_finalize()

    plot_benchmarks(results, D_min, D_max, "Forward solve", "forward_solve")
end

methods = Dict{Val, String}(
    Val(:cpu_mt) => "CPU (multithreaded)",
    Val(:ours) => "Ours",
    Val(:magma_non_strided) => "MAGMA (non-strided)",
    Val(:gpu_mem_bound) => "SOL",
    Val(:cublas_non_strided) => "cuBLAS (non-strided)",
)

generate_plots(2, 15, methods, Float32)