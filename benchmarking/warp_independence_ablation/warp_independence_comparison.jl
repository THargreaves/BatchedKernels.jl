using Magma
using JLD2

include("../plot_benchmarks.jl")
include("../generate_tables.jl")
include("gpu_mem_bound.jl")
include("warp_independent_kalman.jl")
include("synced_kalman.jl")

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
                "kalman_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            # println(cache_file)
            if isfile(cache_file)
                @load cache_file time
                # println("  Using cached result for $label, D = $D: $time s")
            else
                N = Int(ceil(1e9 / (4 * 2 * D^2)))

                P_out = zeros(T, D, D, N)
                P_in = zeros(T, D, D, N)
                for i in 1:N
                    P_i = rand(T, D, D) / T(D)
                    P_i = P_i * P_i' + 0.1f0 * I
                    P_in[:, :, i] = P_i
                end

                A = rand(T, D, D) / Float32(D)

                Q_elem = rand(Float32, D, D) / Float32(D)^2
                Q = Q_elem * Q_elem' + 0.01f0 * I

                H = rand(Float32, D, D) / Float32(D)
                R_elem = rand(Float32, D, D) / Float32(D)^2
                R = R_elem * R_elem' + 0.01f0 * I

                time = kalman_timing(P_out, P_in, A, Q, H, R, queue_ptr, method)

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

    plot_benchmarks(results, D_min, D_max, "Warp independent vs synced", "warp_dependence_ablation")
    write_results_csv(results, D_min, D_max, "warp_dependence_ablation")
end

methods = Dict{Val, String}(
    Val(:independent) => "Warp independent",
    Val(:synced) => "Warp dependent (synced)",
    Val(:gpu_mem_bound) => "Memory bound",
)

generate_plots(2, 16, methods, Float32)