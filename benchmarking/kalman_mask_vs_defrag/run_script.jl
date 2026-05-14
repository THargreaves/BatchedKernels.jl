using BatchedKernels
using LinearAlgebra
using CUDA
using CUDA: i32
using Random
using BenchmarkTools
using Plots
using Printf
using JLD2

include("kalman_mask.jl")
include("kalman_defrag.jl")

function generate_plot(D_max::Integer; redo::Bool = false)
    cache_dir = joinpath(@__DIR__, "cache")
    isdir(cache_dir) || mkdir(cache_dir)
    
    nthreads = 2^8
    xs = collect(2:D_max)
    ys = collect(2:D_max)
    T = Float32

    ratios = Matrix{Float64}(undef, D_max - 1, D_max - 1)

    count = 0
    total_count = (D_max - 1)^2

    for Dx in xs
        for Dy in ys
            cache_file = joinpath(
                cache_dir,
                "kalman_mem_$(Dx)_$(Dy).jld2"
            )

            if isfile(cache_file) && !redo
                @load cache_file ratio
            else
                Random.seed!(1234)

                D = max(Dx, Dy)
                N = Int(ceil(1e9 / (4 * 2 * Dx * Dy)))
                nblocks = cld(N, nthreads//32 * (32 ÷ D))

                P_cpu = Array{T}(undef, Dx, Dx, N)
                for i in 1:N
                    P_i = rand(T, Dx, Dx) / T(Dx)
                    P_i = P_i * P_i' + 0.1f0 * I
                    P_cpu[:, :, i] = P_i
                end
                P_in = CuArray(P_cpu)

                F_cpu = rand(Float32, Dx, Dx) / Float32(Dx)
                F = cu(F_cpu)

                Q_cpu = rand(Float32, Dx, Dx) / Float32(Dx)^2
                Q_cpu = Q_cpu * Q_cpu' + 0.01f0 * I
                Q = cu(Q_cpu)

                H_cpu = rand(T, Dy, Dx) / T(D)
                H = cu(H_cpu)

                R_cpu = rand(T, Dy, Dy) / T(Dy)^2
                R_cpu = R_cpu * R_cpu' + 0.01f0 * I
                R = cu(R_cpu)

                P_out = CUDA.zeros(T, Dx, Dx, N)

                result_curr = @benchmark begin
                    CUDA.@sync @cuda threads=$nthreads blocks=$nblocks $kernel_kalman_mask!(
                        $P_out, $P_in, $F, $Q, $H, $R, Val(Int32($Dx)), Val(Int32($Dy)), Val(Int32($D)), Val(Int32($nthreads)), Int32($N),
                    )
                end

                result_defrag = @benchmark begin
                    CUDA.@sync @cuda threads=$nthreads blocks=$nblocks $kernel_kalman_defrag!(
                        $P_out, $P_in, $F, $Q, $H, $R, Val(Int32($Dx)), Val(Int32($Dy)), Val(Int32($D)), Val(Int32($nthreads)), Int32($N),
                    )
                end

                ratio = median(result_curr.times) / median(result_defrag.times)

                @save cache_file ratio

                CUDA.reclaim()
                GC.gc()
            end
            
            ratios[Dy - 1, Dx - 1] = ratio

            count += 1
            println("$count/$total_count ($Dx,$Dy): ratio=$ratio")
        end
    end

    plt = heatmap(
        xs, ys, ratios;
        size = (600, 500),
        xlabel = "Dx",
        ylabel = "Dy",
        title = "time (mask) / time (defrag)",
        aspect_ratio = :equal,
        colorbar = true,
        interpolate = false,
        xlims = (1.5, D_max + 0.5),
        ylims = (1.5, D_max + 0.5),
        xticks = xs,
        yticks = ys,
    )

    rmin, rmax = extrema(ratios)
    threshold = rmin + 0.35 * (rmax - rmin)

    ann = [(Dx, Dy, text(
        @sprintf("%.2f", ratios[Dy - 1, Dx - 1]),
        9,
        ratios[Dy - 1, Dx - 1] ≤ threshold ? :white : :black,
    )) for Dy in ys for Dx in xs]
    annotate!(plt, ann)

    path = joinpath(dirname(@__DIR__), "figs", "kalman_ratio_heatmap_D_$D_max.svg")
    savefig(plt, path)
    display(plt)
end

generate_plot(11, redo=false)