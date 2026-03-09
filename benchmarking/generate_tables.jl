using DataFrames
using CSV

function write_results_csv(
    results::Dict{String, Vector{Float64}},
    D_min::Int,
    D_max::Int,
    filename::String,
)
    Ds = collect(D_min:D_max)
    nD = length(Ds)

    # Sanity checks
    for (method, runtimes) in results
        length(runtimes) == nD ||
            error("Method '$method' has length $(length(runtimes)), expected $nD")
    end

    # Create DataFrame with D as first column
    df = DataFrame(D = Ds)

    # Add one column per method
    for (method, runtimes) in sort(collect(results); by=first)
        df[!, Symbol(method)] = runtimes
    end

    tables_dir = joinpath(@__DIR__, "tables")
    mkpath(tables_dir)
    path = joinpath(tables_dir, "$filename.csv")

    CSV.write(path, df)

    return nothing
end