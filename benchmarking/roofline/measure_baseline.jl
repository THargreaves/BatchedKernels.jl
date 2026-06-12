"""
measure_baseline.jl

Measurement stage of the baseline roofline pipeline. For a MULTI-KERNEL
baseline implementation (e.g. JAX QR, which wraps a cuSOLVER call in
layout-transpose kernels), the analytic single-pass DRAM-byte count is
NOT valid -- the extra kernels move data extra times. This stage measures
the real DRAM traffic directly from hardware counters.

RELATION TO discover_kernels.jl
  `discover_kernels` (the discovery stage) runs `nsys` and classifies the
  baseline's kernels, yielding the verdict `multi_kernel` (does this
  baseline need a measured-bytes run?) plus `kernels_per_call` as a
  non-binding cross-check. This file `include`s it.

  `measure_baseline` runs discovery (or uses its cached scalars), then
  the `ncu` measurement, then parses. The `ncu` run is cached -- a CSV
  is reused if it parses and contains a marker row.

METHOD
  Run the baseline driver under `ncu` with NO kernel-name filter and NO
  launch skip/count -- capture EVERY kernel. The driver runs MEASURE_L
  calls and emits a uniquely-named no-op MARKER kernel once, right before
  the final call. The steady-state call is then isolated as the kernel
  rows AFTER the last marker occurrence (warmup / JIT / autotuning all
  precede the marker). This needs no launch-offset arithmetic: warmup
  kernel counts are variable and unpredictable, so any skip-N-launches
  scheme is unsound -- the marker is a self-locating call boundary.

  `ncu`'s dram__bytes_read/write.sum hardware counters give the REAL DRAM
  traffic per kernel; summing over the steady-state call's kernel rows
  gives `total_measured` for exactly ONE call.

  Per-batch-element measured bytes (what the roofline x-axis needs):
    measured_bytes_per_batch_elem = total_measured / N
  where N = batch_size(op, D) is the batch size the driver fixed. The
  bytes are MEASURED (hardware counters); N is only the known divisor
  that converts a one-call total into a per-batch-element rate.

TOOLING
  `nsys` (via discover_kernels) and `ncu`. No extra packages; Base only,
  plus flops.jl for batch_size. Paths anchored to this file's directory.
"""

include(joinpath(@__DIR__, "ncu_csv.jl"))         # parse_ncu_csv, UNIT_SCALE, ...
include(joinpath(@__DIR__, "discover_kernels.jl"))
include(joinpath(@__DIR__, "flops.jl"))   # for batch_size(op, D)

# The only ncu metrics this stage needs: total DRAM bytes read + written
# Full profiling takes ages
const DRAM_METRICS = "dram__bytes_read.sum,dram__bytes_write.sum"

# Marker kernel name
const MARKER_TOKEN = "roofline_marker"

# Run `ncu` on the baseline driver. No kernel-name filter and no launch
# skip/count. Capture every kernel the driver launches. The driver runs
# MEASURE_L calls and emits the marker kernel before the final call.
# The steady-state call is then located by the marker in the parsed output
# (see _marker_split), not by fragile launch-offset arithmetic (as was done initially)
#
# Returns the raw-page CSV path. Cached: a cached CSV is reused only if
# it parses and actually contains a marker row, otherwise it is treated
# as stale and re-profiled (force=true always re-profiles.)
function _run_ncu_baseline(op, impl, D, L; profile_dir, force::Bool)
    haskey(COUNT_DRIVERS, (op,impl)) ||
        error("no driver registered for ($op, $impl) -- add to COUNT_DRIVERS")
    drv = COUNT_DRIVERS[(op,impl)]
    isdir(profile_dir) || mkpath(profile_dir)

    rep = joinpath(profile_dir, "$(op)_$(impl)_D$(D)")     # ncu export base
    csv = rep * ".csv"

    # Cache validity is not mere existence
    # A usable CSV must parse AND contain a marker row
    # Otherwise reprofile
    if isfile(csv) && !force && _csv_has_marker(csv)
        println("  measurement CSV cached, reusing: $csv")
        return csv
    end

    # resolve exe + script (anchored to roofline/, like discover_kernels)
    exe = occursin('/', drv.exe) ? _rel(drv.exe) : drv.exe
    script = _rel(drv.script)
    isfile(script) || error("baseline driver script not found: $script")

    drv_cmd = if drv.runner == "julia"
        proj = _rel(JULIA_PROJECT_REL)
        `$exe --project=$proj $script $D $L`
    elseif drv.runner == "python"
        `$exe $script $D $L`
    else
        error("unknown runner '$(drv.runner)'")
    end

    # ncu: no --kernel-name filter, no --launch-skip/--launch-count,
    # capture every kernel. The marker delimits the steady-state call in
    # post-processing, so there is no launch arithmetic to get wrong.
    #
    # --metrics (no --set full): this stage needs only the two DRAM-byte
    # counters. --set full replays each kernel ~38x.
    # For a multi-thousand-kernel baseline that is too slow
    println("  profiling $impl with ncu (all kernels, dram-byte metrics only)...")
    run(`ncu --target-processes all --metrics $DRAM_METRICS
         --export $rep --force-overwrite $drv_cmd`)

    run(pipeline(`ncu --import $(rep * ".ncu-rep") --csv --page raw`, stdout=csv))
    isfile(csv) || error("ncu CSV export failed: $csv")
    return csv
end

# does an ncu CSV parse and contain at least one marker row?
function _csv_has_marker(path)
    try
        rows = parse_ncu_csv(path)
        return any(r -> occursin(MARKER_TOKEN, String(get(r, "Kernel Name", ""))), rows)
    catch
        return false
    end
end

# Locate the marker and return the steady-state call's kernel rows:
# all rows after the marker occurrence (marker excluded).
# The driver emits the marker once, right before the final call, and
# does nothing after that call, so "rows after the last marker" is
# exactly one complete steady-state call, free of warmup/JIT/autotuning.
function _marker_split(rows)
    marker_idx = findlast(
        r -> occursin(MARKER_TOKEN, String(get(r, "Kernel Name", ""))), rows)
    marker_idx === nothing && error(
        "marker kernel ('$MARKER_TOKEN') not found in ncu capture -- " *
        "does the driver launch it before the final iteration?")
    call_rows = rows[(marker_idx + 1):end]
    isempty(call_rows) && error(
        "no kernels captured after the marker -- the driver must run one " *
        "full operation call after emitting the marker, and nothing else")
    return call_rows
end

# Unit-recognition assertion. parse_ncu_csv normalizes via UNIT_SCALE but
# treats an unknown unit as scale 1.0, silently wrong by 1e3..1e9 for a
# byte counter. Re-read the units row directly and require every named
# metric's unit to be in UNIT_SCALE; error loudly otherwise.
function _assert_known_units(csv_path, metric_names)
    raw = readlines(csv_path)
    start = findfirst(l -> startswith(l, "\"ID\""), raw)
    start === nothing && error("no header row in $csv_path")
    (length(raw) >= start + 1) || error("no units row in $csv_path")

    splitcells(l) = begin
        s = l
        startswith(s, "\"") && (s = s[2:end])
        endswith(s, "\"")   && (s = s[1:end-1])
        String.(split(s, "\",\""))
    end
    header = splitcells(raw[start])
    units  = splitcells(raw[start + 1])

    for m in metric_names
        j = findfirst(==(m), header)
        j === nothing && error("metric '$m' not in $csv_path")
        u = strip(get(units, j, ""))
        haskey(UNIT_SCALE, u) ||
            error("UNRECOGNIZED unit '$u' for metric '$m' in $csv_path -- " *
                  "add it to UNIT_SCALE in analysis.jl; measured bytes would " *
                  "otherwise be silently wrong by a factor of 1e3..1e9")
    end
    return nothing
end

# Per-kernel byte breakdown. Group captured rows by kernel name, average
# over the captured calls, express per-batch-element and as a multiple of the
# analytic single-pass count.
function _per_kernel_breakdown(rows, N, analytic_pbe)
    # short, stable kernel-name key: drop template args and arg list
    shortname(nm) = begin
        s = String(nm)
        for cut in ('<', '(')
            i = findfirst(cut, s)
            i === nothing || (s = s[1:i-1])
        end
        strip(s)
    end

    agg = Dict{String,NamedTuple{(:rd,:wr,:count),Tuple{Float64,Float64,Int}}}()
    for r in rows
        nm = shortname(get(r, "Kernel Name", "?"))
        rd = get(r, "dram__bytes_read.sum", 0.0)
        wr = get(r, "dram__bytes_write.sum", 0.0)
        rd isa Number || (rd = 0.0)
        wr isa Number || (wr = 0.0)
        prev = get(agg, nm, (rd=0.0, wr=0.0, count=0))
        agg[nm] = (rd=prev.rd+rd, wr=prev.wr+wr, count=prev.count+1)
    end

    # one entry per distinct kernel, per-batch-elem figures, sorted by total.
    # rows are a single steady-state call, so per-batch-elem = bytes / N.
    out = NamedTuple[]
    for (nm, a) in agg
        rd_pbe  = a.rd / N
        wr_pbe  = a.wr / N
        tot_pbe = rd_pbe + wr_pbe
        push!(out, (
            kernel = nm,
            instances = a.count,
            rd_bytes_per_batch_elem = rd_pbe,
            wr_bytes_per_batch_elem = wr_pbe,
            total_bytes_per_batch_elem = tot_pbe,
            x_analytic = analytic_pbe > 0 ? tot_pbe / analytic_pbe : NaN,
        ))
    end
    return sort(out; by = x -> x.total_bytes_per_batch_elem, rev = true)
end

function _print_kernel_breakdown(breakdown, analytic_pbe)
    println("  per-kernel DRAM (bytes/batch elem, x = multiple of analytic " *
            "$(round(analytic_pbe,digits=1)) B):")
    for k in breakdown
        println("    ", rpad(k.kernel, 34),
                rpad(string(round(k.total_bytes_per_batch_elem, digits=2)), 10),
                "(", round(k.x_analytic, digits=2), "x)")
    end
end

function _write_kernel_breakdown(path, op, impl, D, N, analytic_pbe, breakdown)
    open(path, "w") do io
        println(io, "operation,implementation,D,batch_size," *
                    "analytic_bytes_per_batch_elem,kernel,instances," *
                    "rd_bytes_per_batch_elem,wr_bytes_per_batch_elem," *
                    "total_bytes_per_batch_elem,x_analytic")
        for k in breakdown
            println(io, join((op, impl, D, N, analytic_pbe,
                "\"$(k.kernel)\"", k.instances,
                k.rd_bytes_per_batch_elem, k.wr_bytes_per_batch_elem,
                k.total_bytes_per_batch_elem, k.x_analytic), ","))
        end
    end
end

# measure_baseline(op, impl, D)
#
# End-to-end: discover (or reuse cached discovery) ->
# ncu measurement -> parse -> per-batch-element measured DRAM bytes.
#
# Returns a NamedTuple including measured_bytes_per_batch_elem, which the
# roofline uses as the x-axis byte count for a non-fused baseline in
# place of the analytic single-pass value.
function measure_baseline(op::AbstractString, impl::AbstractString, D::Int;
                          force::Bool=false,
                          profile_dir::AbstractString=_rel(joinpath(op, "profile_results")))

    println("=== measure_baseline: $op / $impl  (D=$D) ===")

    # discovery (cached): need the multi_kernel verdict
    # discover_kernels writes <op>_<impl>_D<d>_discovery.csv with the scalars
    scalar_csv = joinpath(profile_dir, "$(op)_$(impl)_D$(D)_discovery.csv")
    disc = if isfile(scalar_csv) && !force
        println("  discovery cached, reading $scalar_csv")
        _read_discovery_cache(scalar_csv)
    else
        println("  running discovery...")
        discover_kernels(op, impl; D=D, profile_dir=profile_dir)
    end

    disc.multi_kernel || @warn "($op,$impl) is single-kernel, analytic " *
        "DRAM bytes are valid, a measured-bytes run is normally unnecessary"

    # ncu measurement (cached)
    # The driver runs MEASURE_L calls and emits the marker kernel before
    # the final call. ncu captures EVERY kernel.
    # The steady-state call is isolated below by _marker_split (rows after the marker).
    csv = _run_ncu_baseline(op, impl, D, MEASURE_L;
                            profile_dir=profile_dir, force=force)
    all_rows = parse_ncu_csv(csv)        # from ncu_csv.jl, one row per kernel

    # Unit-recognition assertion
    # parse_ncu_csv normalizes units via UNIT_SCALE, but an unrecognised
    # unit is silently treated as scale 1.0, which for a byte counter
    # would corrupt the total by 1e3..1e9. ncu auto-scales per file, and
    # read/write can even come in DIFFERENT units. Verify before trusting.
    _assert_known_units(csv, ["dram__bytes_read.sum", "dram__bytes_write.sum"])

    # Isolate the single steady-state call via the marker
    # Rows after the last marker occurrence = exactly one complete
    # steady-state call (warmup/JIT/autotuning all precede the marker).
    rows = _marker_split(all_rows)
    n_kernels_captured = length(rows)

    # Cross-check against discovery's kernels-per-call. This is no longer
    # load-bearing (the marker, not K, delimits the call), it only
    # flags a surprise, e.g. the last call differing from steady state.
    if disc.kernels_per_call > 0 && n_kernels_captured != disc.kernels_per_call
        @warn "marker-delimited call has $n_kernels_captured kernels, " *
              "discovery's kernels-per-call was $(disc.kernels_per_call) -- " *
              "minor mismatch is benign (rounding/classification); a large " *
              "one may mean the last call is not steady-state"
    end

    # Sum the DRAM hardware counters over the steady-state call
    rd = _sum_metric(rows, "dram__bytes_read.sum")
    wr = _sum_metric(rows, "dram__bytes_write.sum")
    (rd isa Number && wr isa Number) ||
        error("dram__bytes_read/write.sum missing from $csv")
    total_measured = rd + wr             # bytes for ONE steady-state call

    # Convert to per-batch-element (1/N)
    # total_measured is exactly one call (the marker delimits one call),
    # and one call processes N batch elements -> divide by N. No capture-calls
    # factor, the marker isolates exactly one call by construction.
    N = batch_size(op, D)
    measured_bytes_per_batch_elem = total_measured / N

    # Per-kernel byte breakdown
    # Group the steady-state call's kernels by name, express per-batch-elem
    # and as a multiple of analytic. Makes the aggregate auditable: shows
    # exactly which kernel moves how much.
    analytic_pbe = dram_bytes_per_batch_elem(op, D)
    breakdown = _per_kernel_breakdown(rows, N, analytic_pbe)
    bd_csv = joinpath(profile_dir, "$(op)_$(impl)_D$(D)_kernel_bytes.csv")
    _write_kernel_breakdown(bd_csv, op, impl, D, N, analytic_pbe, breakdown)

    println("  total measured DRAM (one steady-state call): " *
            "$(round(total_measured/1e9, digits=3)) GB")
    println("  N = $N batch elems/call  ->  measured bytes/batch elem = " *
            "$(round(measured_bytes_per_batch_elem, digits=2)) " *
            "($(round(measured_bytes_per_batch_elem/analytic_pbe, digits=2))x analytic)")
    _print_kernel_breakdown(breakdown, analytic_pbe)
    println("  wrote $bd_csv")

    return (
        operation = op,
        implementation = impl,
        D = D,
        total_measured_dram = total_measured,
        batch_size = N,
        n_kernels_captured = n_kernels_captured,
        measured_bytes_per_batch_elem = measured_bytes_per_batch_elem,
    )
end

# Read the discovery scalars (multi_kernel, kernels_per_call)
function _read_discovery_cache(scalar_csv)
    isfile(scalar_csv) ||
        error("discovery scalar CSV not found: $scalar_csv -- run discover_kernels first")
    lines = readlines(scalar_csv)
    length(lines) < 2 && error("empty/short discovery CSV: $scalar_csv")
    header = _csv_cells(lines[1])
    row    = _csv_cells(lines[2])
    col(name) = begin
        i = findfirst(==(name), header)
        i === nothing && error("column '$name' missing in $scalar_csv")
        strip(row[i])
    end
    return (
        kernels_per_call = parse(Int, col("kernels_per_call")),
        multi_kernel     = parse(Bool, col("multi_kernel")),
    )
end
