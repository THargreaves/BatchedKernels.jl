using Plots

const ROOFLINE_DIR = @__DIR__
_rel(p::AbstractString) = abspath(joinpath(ROOFLINE_DIR, p))

const DRAM_BW_BYTES_PER_S = 1008e9       # 1008 GB/s
const FP32_PEAK_FLOPS     = 82.58e12     # 82.58 TFLOPS FP32 (FFMA peak)

const SHMEM_BW_FILE   = _rel(joinpath("hardware", "shmem_bandwidth.csv"))
const SHMEM_BW_SCRIPT = _rel(joinpath("hardware", "measure_shmem_bw.jl"))

# parse the bandwidth (column 2) out of the one data row of the cache file
function _read_shmem_bw_file()
    isfile(SHMEM_BW_FILE) || return nothing
    lines = readlines(SHMEM_BW_FILE)
    length(lines) < 2 && return nothing
    f = split(strip(lines[2]), ",")
    length(f) < 2 && return nothing
    bw = tryparse(Float64, strip(f[2]))
    (bw === nothing || bw <= 0) && return nothing
    return bw
end

"""
Returns the sustained shared memory bandwidth (bytes/s) as a Float64
by running the microbenchmark kernel
"""
function shmem_bandwidth(; force::Bool=false)
    if !force
        bw = _read_shmem_bw_file()
        bw === nothing || return bw
    end
    isfile(SHMEM_BW_SCRIPT) ||
        error("shared-memory microbenchmark not found: $SHMEM_BW_SCRIPT")
    @info "running shared-memory bandwidth microbenchmark (force=$force)"
    run(`julia --project=$(_rel("../..")) $SHMEM_BW_SCRIPT`)
    bw = _read_shmem_bw_file()
    bw === nothing &&
        error("microbenchmark ran but $SHMEM_BW_FILE is missing/invalid")
    return bw
end

const PLOT_IMPLS = ["ours", "cublas", "cusolver", "jax", "magma"]

const DISPLAY_NAME = Dict(
    "ours"   => "This",
    "cublas" => "cuBLAS",
    "cusolver" => "cuSOLVER",
    "cublas_cusolver" => "cuBLAS/cuSOLVER",
    "jax"    => "JAX (vmap)",
    "magma"  => "MAGMA",
)

# Read one analysis_<op>_<impl>.csv (written by analyse()). Returns the
# roofline columns: D, arithmetic intensity, achieved FLOP/s, and the
# byte source ("analytic"/"measured"). `ai_col` selects which intensity
# column to read: "dram_AI" for the DRAM roofline, "shmem_AI" for the
# shared-memory roofline. Rows whose ai_col is blank/missing are skipped
# (e.g. baselines have no shmem_AI). Returns `nothing` if the file is
# absent, or if no rows have the requested AI column.
function _read_analysis_csv(op::AbstractString, impl::AbstractString;
                            ai_col::AbstractString = "dram_AI",
                            profile::AbstractString = "baseline")
    path = _rel(joinpath(op, "profile_results", profile,
                         "analysis_$(op)_$(impl).csv"))
    isfile(path) || return nothing

    lines = readlines(path)
    length(lines) < 2 && error("analysis CSV has no data rows: $path")
    header = String.(split(strip(lines[1]), ","))
    idx(name) = begin
        i = findfirst(==(name), header)
        i === nothing && error("column '$name' missing in $path " *
                                "-- is analyse() up to date?")
        i
    end
    ci_D   = idx("D")
    ci_ai  = idx(ai_col)
    ci_y   = idx("achieved_flops")
    ci_src = idx("dram_bytes_source")

    Ds  = Int[]; xs = Float64[]; ys = Float64[]; srcs = String[]
    for li in 2:length(lines)
        isempty(strip(lines[li])) && continue
        f = String.(split(strip(lines[li]), ","))
        length(f) < length(header) && continue
        aistr = strip(f[ci_ai])
        # skip rows with no value in the requested AI column (missing is
        # written blank by _csv) -- e.g. baselines have no shmem_AI
        (isempty(aistr) || aistr == "missing") && continue
        push!(Ds,   parse(Int, strip(f[ci_D])))
        push!(xs,   parse(Float64, aistr))
        push!(ys,   parse(Float64, strip(f[ci_y])))
        push!(srcs, strip(f[ci_src]))
    end
    isempty(Ds) && return nothing
    return (D=Ds, ai=xs, flops=ys, src=srcs, path=path)
end

# Gather every implementation's series for an operation.
function _gather_series(op::AbstractString; ai_col::AbstractString = "dram_AI",
                        profile::AbstractString = "baseline")
    series = Dict{String,NamedTuple}()
    for impl in PLOT_IMPLS
        s = _read_analysis_csv(op, impl; ai_col = ai_col, profile = profile)
        s === nothing && continue
        series[impl] = s
    end
    isempty(series) && error("no analysis_$(op)_*.csv files found in " *
        _rel(joinpath(op, "profile_results", profile)) *
        ", run analyse(\"$op\", impl, Ds) for each implementation first")
    return series
end

# x is shared across implementations only if no series used measured
# bytes, as then their x-axes (FLOPS/B) only depend on D, all having the same x-value
# on the roofline, only their FLOPS/s differs
_shared_x(series) = !any(s -> any(==("measured"), s.src), values(series))

# middle D, rounded down to the nearest even number
function _middle_even_D(Dvec)
    mid = Dvec[cld(length(Dvec), 2)]
    return 2 * (mid ÷ 2)
end

function plot_roofline(op::AbstractString;
                       annotate_Ds = nothing,
                       profile::AbstractString = "baseline",
                       outdir::AbstractString = _rel(joinpath(op, "figs", profile)),
                       markershapes = [:circle, :square, :diamond, :utriangle])

    series = _gather_series(op; profile = profile)

    # union of D values present across all series, sorted
    Dvec = sort(unique(vcat((collect(s.D) for s in values(series))...)))

    if annotate_Ds === nothing
        annotate_Ds = unique([first(Dvec), last(Dvec)])
    end
    annset = Set(annotate_Ds)
    shared = _shared_x(series)

    all_x = vcat((collect(s.ai)    for s in values(series))...)
    all_y = vcat((collect(s.flops) for s in values(series))...)
    ridge = FP32_PEAK_FLOPS / DRAM_BW_BYTES_PER_S

    xlo = min(minimum(all_x) / 2, ridge / 4)
    xhi = max(maximum(all_x) * 2, ridge * 10)
    ylo = minimum(all_y) / 3
    yhi = FP32_PEAK_FLOPS * 1.8

    xline = exp10.(range(log10(xlo), log10(xhi); length = 400))
    yline = min.(FP32_PEAK_FLOPS, DRAM_BW_BYTES_PER_S .* xline)

    plt = plot(xscale = :log10, yscale = :log10,
               xlims = (xlo, xhi), ylims = (ylo, yhi),
               xlabel = "Arithmetic Intensity  (FLOP / B)",
               ylabel = "Performance  (FLOP / s)",
            #    title  = "DRAM roofline - $op",
               legend = :bottomright, framestyle = :box,
               minorgrid = true, gridalpha = 0.2, minorgridalpha = 0.1,
               size = (600, 400), dpi = 200)

    plot!(plt, xline, yline; color = :black, lw = 2.5,
          label = "Roofline")
    hline!(plt, [FP32_PEAK_FLOPS]; color = :gray, ls = :dot, lw = 2, label="")
        #    label = "FP32 peak ($(round(FP32_PEAK_FLOPS/1e12, digits=1)) TFLOP/s)")
    vline!(plt, [ridge]; color = :gray, ls = :dot, lw = 2, label = "")

    # shared-x: faint D guide lines first, so trajectories overlay
    if shared && haskey(series, "ours")
        s = series["ours"]
        for (x, D) in zip(s.ai, s.D)
            D in annset || continue
            plot!(plt, [x, x], [ylo, yhi]; color = :gray, alpha = 0.18,
                  lw = 1, label = "")
        end
    end

    # trajectories (sorted by D so the line is monotone)
    for (i, impl) in enumerate(PLOT_IMPLS)
        haskey(series, impl) || continue
        s = series[impl]
        perm = sortperm(collect(s.D))
        xs, ys, ds = s.ai[perm], s.flops[perm], s.D[perm]
        isempty(xs) && continue
        plot!(plt, xs, ys; marker = markershapes[mod1(i, end)],
              ms = 3, markerstrokewidth = 0.4, lw = 1.8,
              label = get(DISPLAY_NAME, impl, impl))

        # non-shared-x: per-series D labels at the points
        # if !shared
        #     for (x, y, D) in zip(xs, ys, ds)
        #         D in annset && annotate!(plt, x, y * 1.28,
        #                                  text("D=$D", 15, :left))
        #     end
        # end
    end

    # shared-x: D labels in a strip along the bottom
    if shared && haskey(series, "ours")
        s = series["ours"]
        ylabel_pos = ylo * 1.5
        for (x, D) in zip(s.ai, s.D)
            D in annset || continue
            vline!(plt, [x]; color = :gray, ls = :dot, lw = 1.5, label = "")
            annotate!(plt, x, ylabel_pos, text("D=$D", 10, :center, :black))
        end
    end

    mkpath(outdir)
    outpath = joinpath(outdir, "roofline_$(op).png")
    savefig(plt, outpath)
    println("wrote $outpath")
    return plt
end

function plot_shmem_roofline(op::AbstractString;
                             annotate_Ds = nothing,
                             profile::AbstractString = "baseline",
                             outdir::AbstractString = _rel(joinpath(op, "figs", profile)))

    series = _gather_series(op; ai_col = "shmem_AI", profile = profile)

    # measured shared-memory ceiling, from the microbenchmark cache. The
    # pipeline primes/refreshes it (run_pipeline.sh)
    shmem_bw = shmem_bandwidth()

    Dvec = sort(unique(vcat((collect(s.D) for s in values(series))...)))
    if annotate_Ds === nothing
        annotate_Ds = unique([first(Dvec), last(Dvec)])
    end
    annset = Set(annotate_Ds)

    all_x = vcat((collect(s.ai)    for s in values(series))...)
    all_y = vcat((collect(s.flops) for s in values(series))...)
    ridge = FP32_PEAK_FLOPS / shmem_bw

    xlo = min(minimum(all_x) / 2, ridge / 4)
    xhi = max(maximum(all_x) * 2, ridge * 10)
    ylo = minimum(all_y) / 3
    yhi = FP32_PEAK_FLOPS * 1.8

    xline = exp10.(range(log10(xlo), log10(xhi); length = 400))
    yline = min.(FP32_PEAK_FLOPS, shmem_bw .* xline)

    plt = plot(xscale = :log10, yscale = :log10,
               xlims = (xlo, xhi), ylims = (ylo, yhi),
               xlabel = "Arithmetic Intensity  (FLOP / shared-memory B)",
               ylabel = "Performance  (FLOP / s)",
            #    title  = "Shared-memory roofline - $op",
               legend = :bottomright, framestyle = :box,
               minorgrid = true, gridalpha = 0.2, minorgridalpha = 0.1,
               size = (600, 400), dpi = 200)

    plot!(plt, xline, yline; color = :black, lw = 2.5,
          label = "Shared memory roofline")# (shared-mem BW $(round(shmem_bw/1e12, digits=1)) TB/s / FP32 peak)")
    hline!(plt, [FP32_PEAK_FLOPS]; color = :gray, ls = :dot, lw = 2, label="",)
        #    label = "FP32 peak ($(round(FP32_PEAK_FLOPS/1e12, digits=1)) TFLOP/s)")
    vline!(plt, [ridge]; color = :gray, ls = :dot, lw = 2, label = "")

    # trajectory (sorted by D), D labels next to each point
    for (i, impl) in enumerate(PLOT_IMPLS)
        haskey(series, impl) || continue
        s = series[impl]
        perm = sortperm(collect(s.D))
        xs, ys, ds = s.ai[perm], s.flops[perm], s.D[perm]
        isempty(xs) && continue
        plot!(plt, xs, ys; marker = :circle, ms = 4,
              markerstrokewidth = 0.4, lw = 1.8,
              label = get(DISPLAY_NAME, impl, impl))
        # for (x, y, D) in zip(xs, ys, ds)
        #     D in annset && annotate!(plt, x, y * 1.28, text("D=$D", 9, :left))
        # end
    end

    mkpath(outdir)
    outpath = joinpath(outdir, "shmem_roofline_$(op).png")
    savefig(plt, outpath)
    println("wrote $outpath")
    return plt
end
