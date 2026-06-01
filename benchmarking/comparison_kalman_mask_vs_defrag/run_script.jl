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
include("../config/Schedule.jl")

function write_ratios_csv(path::AbstractString,
  ratios::AbstractMatrix; D_min::Int = 2)
      nDy, nDx = size(ratios)
      open(path, "w") do io
          println(io, "Dx,Dy,ratio")
          for j in 1:nDy
              Dy = D_min + j - 1
              for i in 1:nDx
                  Dx = D_min + i - 1
                  println(io, "$Dx,$Dy,$(ratios[j, i])")
              end
          end
      end
      println("Wrote $(nDx * nDy) rows to $path")
  end

function generate_plot(D_max::Integer, force::Bool)
    cache_dir = joinpath(@__DIR__, "cache")
    isdir(cache_dir) || mkdir(cache_dir)
    
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

            if !force && isfile(cache_file)
                @load cache_file ratio
            else
                Random.seed!(1234)
                nthreads = 128  # Limited by the shmem buffers in defrag
                D = max(Dx, Dy)
                N = Int(ceil(1e9 / (4 * 2 * Dx * Dy)))

                P_in = Array{T}(undef, Dx, Dx, N)
                for i in 1:N
                    P_i = rand(T, Dx, Dx) / T(Dx)
                    P_i = P_i * P_i' + 0.1f0 * I
                    P_in[:, :, i] = P_i
                end

                A_cpu = rand(Float32, Dx, Dx) / Float32(Dx)

                Q_cpu = rand(Float32, Dx, Dx) / Float32(Dx)^2
                Q_cpu = Q_cpu * Q_cpu' + 0.01f0 * I

                H_cpu = rand(T, Dy, Dx) / T(D)

                R_cpu = rand(T, Dy, Dy) / T(Dy)^2
                R_cpu = R_cpu * R_cpu' + 0.01f0 * I

                P_out = zeros(T, Dx, Dx, N)

                median_curr = kalman_timing(P_out, P_in, A_cpu, Q_cpu, H_cpu, R_cpu, nthreads, Val(:mask))
                median_defrag = kalman_timing(P_out, P_in, A_cpu, Q_cpu, H_cpu, R_cpu, nthreads, Val(:defrag))

                ratio = median_curr / median_defrag

                @save cache_file ratio

                CUDA.reclaim()
                GC.gc()
            end
            
            ratios[Dy - 1, Dx - 1] = ratio

            count += 1
            println("$count/$total_count ($Dx,$Dy): ratio=$ratio")
        end
    end

    # plt = heatmap(
    #     xs, ys, ratios;
    #     size = (1200, 1200),
    #     xlabel = "Dx",
    #     ylabel = "Dy",
    #     title = "time (mask) / time (defrag)",
    #     aspect_ratio = :equal,
    #     colorbar = true,
    #     interpolate = true,
    #     xlims = (1.5, D_max + 0.5),
    #     ylims = (1.5, D_max + 0.5),
    #     # xticks = xs,
    #     # yticks = ys,
    # )
    clims = (0.49, 1.5)
    plt = contourf(
        xs, ys, ratios;
        size = (400, 350),
        xlabel = "Dx",
        ylabel = "Dy",
        title = "",
        aspect_ratio = :equal,
        clims = clims,
        c = :viridis,
        colorbar = true,
        levels = 50,
        linewidth = 0,
        xlims = (1.5, D_max + 0.5),
        ylims = (1.5, D_max + 0.5),
    )

    # rmin, rmax = extrema(ratios)
    # threshold = rmin + 0.35 * (rmax - rmin)
    # ann = [(Dx, Dy, text(
    #     @sprintf("%.2f", ratios[Dy - 1, Dx - 1]),
    #     9,
    #     ratios[Dy - 1, Dx - 1] ≤ threshold ? :white : :black,
    # )) for Dy in ys for Dx in xs]
    # annotate!(plt, ann)

    base = joinpath(dirname(@__DIR__), "comparison_kalman_mask_vs_defrag")
    fig_path = joinpath(base, "figs", "comparison_kalman_mask_vs_defrag_D_$D_max.png")
    table_path = joinpath(base, "tables", "comparison_kalman_mask_vs_defrag_D_$D_max.csv")
    write_ratios_csv(table_path, ratios)
    savefig(plt, fig_path)
    display(plt)
end

function main(force::Bool)
    generate_plot(32, force)
end

force = length(ARGS) >= 1 && ARGS[1] == "force"
main(force)