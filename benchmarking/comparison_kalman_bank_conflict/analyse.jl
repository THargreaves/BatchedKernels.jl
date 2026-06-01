#!/usr/bin/env julia
# ======================================================================
# analyze_dual_ablation.jl
#
# Per-D summary of the dual-access vs naive ablation.
#
# Reads ncu raw page CSVs ({orig,conflict}_<D>.csv) and ncu source page
# CSVs ({orig,conflict}_<D>_sass.csv) from a directory and emits a
# single summary CSV with one row per (D, layout).
#
# Columns produced:
#   D, layout, gcd_D_32,
#   time_ms,
#   total_inst,                  -- warp-instructions executed
#   thread_inst,                 -- thread-instructions executed
#   sm_throughput_pct,           -- sm__throughput.avg.pct_of_peak_sustained_elapsed
#   dram_throughput_pct,
#   eligible_warps_per_cycle,
#   shmem_kb_per_block,
#   registers_per_thread,
#   wave_actual,
#   wave_ideal,
#   wave_excess,
#   excess_fraction,             -- wave_excess / wave_ideal (the SASS-verified
#                                   bank-conflict metric)
#   lds_scalar,
#   lds_64,
#   lds_128,
#   sts_scalar,
#   sts_64,
#   sts_128,
#   wide_load_float_fraction,    -- (4*LDS128 + 2*LDS64) /
#                                   (LDS_scalar + 2*LDS64 + 4*LDS128)
#   ns_per_inst                  -- time_ms*1e6 / total_inst
#
# Usage:
#   julia analyze_dual_ablation.jl <profile_dir> <D_list> [out_csv]
# e.g.
#   julia analyze_dual_ablation.jl /mnt/user-data/uploads "8 10 16 17 18 27" \
#         dual_ablation_summary.csv
# ======================================================================

using CSV, DataFrames

const RAW_KEYS = Dict(
    "time_ms"                   => "gpu__time_duration.sum",
    "sm_throughput_pct"         => "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct"       => "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "eligible_warps_per_cycle"  => "smsp__warps_eligible.avg.per_cycle_active",
    "shmem_kb_per_block"        => "launch__shared_mem_per_block",
    "registers_per_thread"      => "launch__registers_per_thread",
)

parsenum(s::AbstractString) = isempty(s) ? 0.0 : parse(Float64, replace(s, "," => ""))
opcode(src::AbstractString) = let s = strip(src); isempty(s) ? "" : split(s)[1] end

# Read the ncu --page raw CSV. The file has:
#   line 1: column header (quoted)
#   line 2: units row (mostly empty)
#   line 3: data row
# CSV.File correctly handles QUOTED FIELDS CONTAINING COMMAS, which ncu
# emits liberally (kernel-name template parameters contain bare commas
# inside the quoted Kernel Name field). A naive split(",") shifts every
# value left and produces garbage; do not use it.
#
# We read with header=1 so line 1 becomes the column header. Line 2 (the
# units row) then becomes the first data row in the DataFrame, and the
# real data row is the second. The kernel under --launch-count 1 emits
# exactly one data row, so we expect 2 rows in the resulting DataFrame.
function load_raw(path)
    df = CSV.File(path; header=1, types=String, silencewarnings=true) |> DataFrame
    nrow(df) >= 2 || error("load_raw($path): expected >=2 rows (units+data), got $(nrow(df))")
    # second row is the data row; first is units. Build a name->value Dict.
    return Dict(string(name) => something(df[2, name], "") for name in propertynames(df))
end

# Read the ncu --page source --print-source sass CSV. Row 1 is a banner
# row (kernel name). Row 2 is the header. Rows 3..end are per-instruction
# entries. Aggregates:
#   - total Instructions Executed
#   - per-opcode Instructions Executed
#   - wave_actual / wave_ideal / wave_excess summed over Address Space=Shared
function load_sass(path)
    rows = readlines(path)
    # Split each row on comma respecting simple quoted-field handling
    function split_row(s)
        # Simple state-machine CSV: respects double-quoted commas
        out = String[]; buf = IOBuffer(); inq = false
        for c in s
            if c == '"'
                inq = !inq
            elseif c == ',' && !inq
                push!(out, String(take!(buf)))
            else
                print(buf, c)
            end
        end
        push!(out, String(take!(buf)))
        return out
    end
    parts = split_row.(rows)
    header = parts[2]
    iSrc = findfirst(==("Source"), header)::Int
    iIE  = findfirst(==("Instructions Executed"), header)::Int
    iTI  = findfirst(==("Thread Instructions Executed"), header)::Int
    iAS  = findfirst(==("Address Space"), header)::Int
    iWa  = findfirst(==("L1 Wavefronts Shared"), header)::Int
    iWi  = findfirst(==("L1 Wavefronts Shared Ideal"), header)::Int
    iWe  = findfirst(==("L1 Wavefronts Shared Excessive"), header)::Int

    total_inst = 0.0
    thread_inst = 0.0
    op_inst = Dict{String,Float64}()
    wave_actual = wave_ideal = wave_excess = 0.0

    for r in parts[3:end]
        length(r) == length(header) || continue
        isempty(r[1]) && continue
        ie = parsenum(r[iIE]); ti = parsenum(r[iTI])
        total_inst  += ie
        thread_inst += ti
        op = opcode(r[iSrc])
        op_inst[op] = get(op_inst, op, 0.0) + ie
        if r[iAS] == "Shared"
            wave_actual += parsenum(r[iWa])
            wave_ideal  += parsenum(r[iWi])
            wave_excess += parsenum(r[iWe])
        end
    end
    return (; total_inst, thread_inst, wave_actual, wave_ideal, wave_excess, op_inst)
end

# Public-API helper: gather one (D, layout) row.
function gather_one(profile_dir, layout, D)
    raw_path  = joinpath(profile_dir, "$(layout)_$(D).csv")
    sass_path = joinpath(profile_dir, "$(layout)_$(D)_sass.csv")
    isfile(raw_path)  || (@warn "missing $raw_path";  return nothing)
    isfile(sass_path) || (@warn "missing $sass_path"; return nothing)

    raw = load_raw(raw_path)
    sass = load_sass(sass_path)

    # gcd(D, 32)
    g = gcd(D, 32)

    # Pull metrics by ncu name; ncu sometimes returns "n/a" or empty
    metrics = Dict{String,Any}()
    for (k, ncu_k) in RAW_KEYS
        v = get(raw, ncu_k, "")
        metrics[k] = try parsenum(v) catch; missing end
    end

    # Per-opcode counts (default 0 for missing keys)
    op(x) = get(sass.op_inst, x, 0.0)
    lds_scalar = op("LDS") + op("LDS.32")     # ncu may emit LDS or LDS.32 for scalar
    lds_64     = op("LDS.64")
    lds_128    = op("LDS.128")
    sts_scalar = op("STS") + op("STS.32")
    sts_64     = op("STS.64")
    sts_128    = op("STS.128")

    wide_loads_floats  = 4*lds_128 + 2*lds_64
    total_load_floats  = lds_scalar + 2*lds_64 + 4*lds_128
    wide_load_fraction = total_load_floats > 0 ? wide_loads_floats / total_load_floats : 0.0

    excess_fraction = sass.wave_ideal > 0 ? sass.wave_excess / sass.wave_ideal : 0.0

    ns_per_inst = (metrics["time_ms"] !== missing && sass.total_inst > 0) ?
        metrics["time_ms"] * 1e6 / sass.total_inst : missing

    return (
        D = D,
        layout = layout,
        gcd_D_32 = g,
        time_ms = metrics["time_ms"],
        total_inst = sass.total_inst,
        thread_inst = sass.thread_inst,
        sm_throughput_pct = metrics["sm_throughput_pct"],
        dram_throughput_pct = metrics["dram_throughput_pct"],
        eligible_warps_per_cycle = metrics["eligible_warps_per_cycle"],
        shmem_kb_per_block = metrics["shmem_kb_per_block"],
        registers_per_thread = metrics["registers_per_thread"],
        wave_actual = sass.wave_actual,
        wave_ideal = sass.wave_ideal,
        wave_excess = sass.wave_excess,
        excess_fraction = excess_fraction,
        lds_scalar = lds_scalar,
        lds_64 = lds_64,
        lds_128 = lds_128,
        sts_scalar = sts_scalar,
        sts_64 = sts_64,
        sts_128 = sts_128,
        wide_load_float_fraction = wide_load_fraction,
        ns_per_inst = ns_per_inst,
    )
end

function main()
    profile_dir = "profiles"
    Ds = collect(2:32)
    out_csv = "comparison_kalman_bank_conflict.csv"

    rows = NamedTuple[]
    for D in Ds, layout in ("conflict", "orig")
        row = gather_one(profile_dir, layout, D)
        row === nothing && continue
        push!(rows, row)
    end
    isempty(rows) && (println("no rows"); exit(1))
    df = DataFrame(rows)

    # head-to-head ratio columns (dual / naive) when both layouts present
    # Computed in a wide form for human reading; CSV stays long.
    println("\n=== per-(D, layout) summary ===")
    show(stdout, df; allrows = true, allcols = true)
    println()

    println("\n=== head-to-head ratios where available ===")
    for D in Ds
        naive = filter(r -> r.D == D && r.layout == "conflict", df)
        dual  = filter(r -> r.D == D && r.layout == "orig",     df)
        if nrow(naive) == 1 && nrow(dual) == 1
            n = naive[1, :]; d = dual[1, :]
            time_ratio = d.time_ms / n.time_ms
            inst_ratio = d.total_inst / n.total_inst
            println("  D=$D  dual/naive  time = $(round(time_ratio, digits=3))  " *
                    "inst = $(round(inst_ratio, digits=3))  " *
                    "naive_excess = $(round(100*n.excess_fraction, digits=1))%  " *
                    "naive_wide_loads = $(round(100*n.wide_load_float_fraction, digits=1))%  " *
                    "dual_wide_loads = $(round(100*d.wide_load_float_fraction, digits=1))%")
        else
            println("  D=$D  incomplete pair (have: $(unique(filter(r -> r.D == D, df).layout)))")
        end
    end

    CSV.write(out_csv, df)
    println("\nwrote $out_csv  ($(nrow(df)) rows)")
end

main()