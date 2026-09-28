# Run with this directory as the active project. No target-package files are edited.
using Pkg
Base.active_project() == joinpath(@__DIR__, "Project.toml") ||
    error("Activate integration/generalised_filters before running setup.jl")
length(ARGS) in (1, 2) ||
    error("Usage: setup.jl /path/to/GeneralisedFilters [/path/to/SSMProblems]")
packages = [
    PackageSpec(; path=normpath(joinpath(@__DIR__, "../.."))),
    PackageSpec(; path=abspath(ARGS[1])),
]
length(ARGS) >= 2 && push!(packages, PackageSpec(; path=abspath(ARGS[2])))
Pkg.develop(packages)
Pkg.instantiate()
