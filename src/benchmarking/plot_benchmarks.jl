using Plots


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

    plt = plot(
        xlabel = xlabel,
        ylabel = ylabel,
        title  = title,
        legend = :topleft,
        x_scale = :log2,
        y_scale = :log10,
    )

    for (label, result) in results
        plot!(plt, Ds, result, label = label)
    end

    display(plt)
    
    path = joinpath(@__DIR__, "$filename.svg")
    savefig(plt, path)
end