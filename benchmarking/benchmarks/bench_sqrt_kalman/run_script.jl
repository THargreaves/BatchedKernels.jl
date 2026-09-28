using Magma
using JLD2
using CUDA
using BenchmarkTools
using LinearAlgebra
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
                Random.seed!(1234)
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
                Q_elem = Q_elem * Q_elem' + 0.01f0 * I
                S_Q = Float32.(Matrix(cholesky(Q_elem).L))

                H = rand(Float32, D, D) / Float32(D)

                R_elem = rand(Float32, D, D) / Float32(D)^2
                R_elem = R_elem * R_elem' + 0.01f0 * I
                S_R = Float32.(Matrix(cholesky(R_elem).L))

                time = sqrt_kalman_timing(S_out, S_in, A, S_Q, H, S_R, queue_ptr, Val(THRESH), nthreads, method)

                @save cache_file time
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    Magma.LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)
    Magma.LibMagma.magma_finalize()

    plot_benchmarks(results, D_min, D_max, "", "sqrt_kalman", path)
    write_results_csv(results, D_min, D_max, "sqrt_kalman", path)
end

function main(force::Bool)
    methods = Dict{Val, String}(
        Val(:cpu_mt) => "CPU (multithreaded)",
        Val(:ours) => "This",
        Val(:gpu_mem_bound) => "Memory bound",
        Val(:magma) => "MAGMA",
        Val(:jax_vmap) => "JAX (vmap)",
        Val(:cublas) => "cuBLAS",
    )

    path = "benchmarks/bench_sqrt_kalman"

    generate_plots(2, 32, methods, Float32, path, force)
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)