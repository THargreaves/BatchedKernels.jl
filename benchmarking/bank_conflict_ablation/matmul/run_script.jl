using Magma
using JLD2
using LinearAlgebra
using Random

include("../../plot_benchmarks.jl")
include("../../generate_tables.jl")
include("gpu_mem_bound.jl")
include("original_matmul.jl")
include("conflict_matmul.jl")

function generate_plots(D_min::Integer, D_max::Integer, n_step::Int, methods::Dict{Val, String}, T::Type)
    results =  Dict{String, Vector{Float64}}()

    cache_dir = joinpath(@__DIR__, "cache_$n_step")
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
            else
                N = Int(ceil(1e9 / (4 * 2 * D^2)))

                CUDA.seed!(1234)
                Random.seed!(1234)

                As_cpu = rand(Float32, D, D, N)
                # Bs_cpu = rand(Float32, D, D, N)
                # As_cpu = T.(Matrix(qr(randn(T, D, D)).Q))
                Bs_cpu = zeros(Float32, D, D, N)
                for i in 1:N
                    Bs_cpu[:, :, i] = T.(Matrix(qr(randn(T, D, D)).Q))
                end

                As = cu(As_cpu)
                Bs = cu(Bs_cpu)

                Cs = CUDA.zeros(Float32, D, D, N)

                time = matmul_timing(Cs, As, Bs, n_step, method)

                @save cache_file time

                CUDA.reclaim()
                GC.gc()
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    Magma.LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)
    Magma.LibMagma.magma_finalize()

    plot_benchmarks(results, D_min, D_max, "Dual Access Layout vs Naive Layout", "matmul_bank_conflict_ablation_n_step_$n_step")
    write_results_csv(results, D_min, D_max, "matmul_bank_conflict_ablation_n_step_$n_step")
end

methods = Dict{Val, String}(
    Val(:no_conflict) => "Dual Access Layout",
    Val(:conflict) => "Naive Layout",
    Val(:gpu_mem_bound) => "Memory bound",
)

generate_plots(2, 8, 100, methods, Float32)