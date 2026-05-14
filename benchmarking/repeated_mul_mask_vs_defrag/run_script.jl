using BatchedKernels
using LinearAlgebra
using CUDA
using CUDA: i32
using Random
using JLD2
using Plots
using BenchmarkTools
using Printf

include("repeated_mul_defrag.jl")
include("repeated_mul_mask.jl")

function get_ratios(D::Integer, max_n_muls::Integer; redo::Bool = false)
    cache_dir = joinpath(@__DIR__, "cache")
    isdir(cache_dir) || mkdir(cache_dir)
    nthreads = 2^8

    Ds = collect(2:D)
    T = Float32

    ratios = Matrix{Float64}(undef, D - 1, max_n_muls)
    
    count = 0
    total_count = (D - 1) * max_n_muls

    for D1 in Ds
        Random.seed!(1234)

        N = Int(ceil(1e9 / (4 * 2 * D1^2)))
        nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

        M_in_cpu = rand(T, D1, D1, N)
        M_in = cu(M_in_cpu)

        A_cpu = rand(T, D1, D1)
        A = cu(A_cpu)

        M_out = CUDA.zeros(T, D1, D1, N)

        for n_muls in 1:max_n_muls
            cache_file = joinpath(
                cache_dir,
                "repeated_mul_$(D1)_$(D)_$n_muls.jld2",
            )

            if isfile(cache_file) && !redo
                @load cache_file ratio
            else
                result_mask = @benchmark begin
                    CUDA.@sync @cuda threads=$nthreads blocks=$nblocks $kernel_mul_mask!(
                        $M_out, $M_in, $A, Val(Int32($D1)), Val(Int32($D)), Val(Int32($nthreads)), Val($n_muls), Int32($N),
                    )
                end

                result_defrag = @benchmark begin
                    CUDA.@sync @cuda threads=$nthreads blocks=$nblocks $kernel_mul_defrag!(
                        $M_out, $M_in, $A, Val(Int32($D1)), Val(Int32($D)), Val(Int32($nthreads)), Val($n_muls), Int32($N),
                    )
                end

                ratio = median(result_mask.times) / median(result_defrag.times)

                @save cache_file ratio
            end

            ratios[D1 - 1, n_muls] = ratio

            count += 1
            println("$count/$total_count ($D1,$D) n_muls=$n_muls ratio=$ratio")

            CUDA.reclaim()
            GC.gc()
        end
    end

    return ratios
end

function generate_plots(D::Integer, max_n_muls::Integer; redo::Bool = false)
    ratios = get_ratios(D, max_n_muls; redo = redo)
    xs = 1:max_n_muls
    ys = 2:D

    plt = heatmap(
        xs, ys, ratios;
        size = (2000, 500),
        xlabel = "n_muls",
        ylabel = "D",
        title = "time(mask) / time(defrag)",
        aspect_ratio = :auto,
        colorbar = true,
        interpolate = false,
        xlims = (0.5, max_n_muls + 0.5),
        ylims = (1.5, D + 0.5),
        xticks = xs,
        yticks = ys,
        left_margin = 5Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
        top_margin = 5Plots.mm,
    )

    rmin, rmax = extrema(ratios)
    threshold = rmin + 0.35 * (rmax - rmin)

    ann = [
        (x, y, text(
            @sprintf("%.2f", ratios[y-1, x]),
            9,
            ratios[y-1, x] ≤ threshold ? :white : :black,
        ))
        for y in ys for x in xs
    ]

    annotate!(plt, ann)

    path = joinpath(dirname(@__DIR__), "figs", "repeated_mul_D_$(D)_n_muls_$max_n_muls.svg")
    savefig(plt, path)
    display(plt)
end

generate_plots(8, 40, redo = false)
