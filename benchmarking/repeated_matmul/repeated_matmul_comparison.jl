using Magma
using JLD2

include("../plot_benchmarks.jl")
include("magma_non_strided.jl")
include("cpu_mt.jl")
include("gpu_mem_bound.jl")

function pretty_bytes(n::Integer)
    n < 0 && throw(ArgumentError("byte count must be non-negative"))

    if n < 1_000
        return string(n, "B")
    elseif n < 1_000_000
        return string(round(Int, n / 1_000), "KB")
    elseif n < 1_000_000_000
        return string(round(Int, n / 1_000_000), "MB")
    elseif n < 1_000_000_000_000
        return string(round(Int, n / 1_000_000_000), "GB")
    else
        return string(round(Int, n / 1_000_000_000_000), "TB")
    end
end


function generate_plots(D_min::Integer, D_max::Integer, methods::Dict{Val, String}, T::Type, target_size::Integer)
    results =  Dict{String, Vector{Float64}}()

    target_size_str = pretty_bytes(target_size)

    cache_dir = joinpath(@__DIR__, "cache_$(target_size_str)")
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
                "matmul_repeated_$(string(T))_$(method_sanitised)_D_$(D).jld2",
            )
            if isfile(cache_file)
                @load cache_file time
                # println("  Using cached result for $label, D = $D: $time s")
            else
                N = Int(ceil(target_size / (4 * 2 * D^2)))

                A = rand(T, D, D, N)
                B = rand(T, D, D, N)
                C = similar(A)

                time = repeated_matmul_timing(C, A, B, queue_ptr, method)

                @save cache_file time
            end
            push!(curr_res, time)
        end

        results[label] = curr_res
        println("Finished $label")
    end

    Magma.LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)
    Magma.LibMagma.magma_finalize()

    plot_benchmarks(results, D_min, D_max, "Repeated matmul, target size=$(target_size_str)", "repeated_matmul_$(target_size_str)")
end

methods = Dict{Val, String}(
    Val(:cpu_mt) => "CPU (multithreaded)",
    Val(:magma_non_strided) => "MAGMA (non-strided)",
    Val(:gpu_mem_bound) => "SOL",
)

for i in 1:10
    generate_plots(2, 15, methods, Float32, i * 10^7)
end