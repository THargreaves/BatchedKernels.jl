using Plots

function get_bounds(results::Dict{String, Vector{Float64}})
    all_vals = Iterators.flatten(values(results))

    return minimum(all_vals), maximum(all_vals)
end


function plot_benchmarks(
    results::Dict{String, Vector{Float64}},
    D_min::Integer,
    D_max::Integer,
    title::Union{String,Nothing},
    filename::String,
    path::String;
    xlabel="D",
    ylabel="Time per matrix (s)",
    xscale=:log2,
    yscale=:log10,
    hline=nothing,
    legend=:topleft,
    xlims=:auto,
    ylims=:auto,
)
    Ds = collect(D_min:D_max)

    y_min, y_max = get_bounds(results)

    if xscale == :log2
        xticks = 2 .^ (floor(Int, log2(D_min)):ceil(Int, log2(D_max)))
        if xticks[end] != D_max
            xticks = vcat(xticks, D_max)
        end
        xtick_spec = (xticks, string.(Int.(round.(xticks))))
    else
        xtick_spec = :auto
    end

    if yscale == :log10
        yticks = 10.0 .^ (floor(Int, log10(y_min)):(ceil(Int, log10(y_max)) + 2))
    else
        yticks = :auto
    end

    plt = plot(
        size = (600, 400),
        xlabel = xlabel,
        ylabel = ylabel,
        title  = title,
        legend = legend,
        x_scale = xscale,
        y_scale = yscale,
        xticks = xtick_spec,
        yticks = yticks,
        grid=true,
        minorgrid=true,
        gridalpha=0.3,
        minorgridalpha=0.3,
        xlims=xlims,
        ylims=ylims,
    )

    for (label, result) in sort(collect(results); by = pair -> pair[2][1])
        plot!(plt, Ds, result, linewidth = 2, label = label)
    end

    if hline !== nothing
        Plots.hline!(plt, [hline], linestyle = :dash, linewidth = 2,
                     color = :black, label = "asymptote ($hline)")
    end

    display(plt)

    figs_dir = joinpath(@__DIR__, path, "figs")
    mkpath(figs_dir)

    path = joinpath(figs_dir, "$filename.png")
    savefig(plt, path)
end