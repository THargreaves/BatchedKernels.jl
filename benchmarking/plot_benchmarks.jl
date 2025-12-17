using Plots

function get_bounds(results::Dict{String, Vector{Float64}})
    all_vals = Iterators.flatten(values(results))

    return minimum(all_vals), maximum(all_vals)
end


function plot_benchmarks(
    results::Dict{String, Vector{Float64}},
    D_min::Integer,
    D_max::Integer,
    title::String,
    filename::String;
    xlabel="D",
    ylabel="Time per matrix (s)",
)
    Ds = collect(D_min:D_max)

    y_min, y_max = get_bounds(results)

    plt = plot(
        xlabel = xlabel,
        ylabel = ylabel,
        title  = title,
        legend = :topleft,
        x_scale = :log2,
        y_scale = :log10,
        xticks=2 .^ (floor(Int, log2(D_min)):ceil(Int, log2(D_max))),
        yticks=10.0 .^ (floor(Int, log10(y_min)):(ceil(Int, log10(y_max)) + 2)),
    )

    for (label, result) in results
        plot!(plt, Ds, result, label = label)
    end

    display(plt)
    
    path = joinpath(@__DIR__, "figs", "$filename.svg")
    savefig(plt, path)
end