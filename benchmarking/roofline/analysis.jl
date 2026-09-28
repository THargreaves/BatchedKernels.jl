include(joinpath(@__DIR__, "ncu_csv.jl"))
include("flops.jl")
include("sass_analysis.jl")

const ROOFLINE_DIR = @__DIR__
_rel(p::AbstractString) = abspath(joinpath(ROOFLINE_DIR, p))

include("measure_baseline.jl")

"""
analysis.jl

Turns raw `ncu` profile CSVs + benchmark timing tables into roofline-ready
derived numbers.

Pipeline position:
    profiling (slow, cached) -> analysis.jl (fast, not cached) -> plotting
"""

# Implementation -> benchmark CSV column names
const BENCH_COL = Dict{String,Vector{String}}(
    "ours"     => ["This"],
    "cublas"   => ["cuBLAS"],
    "cusolver" => ["cuSOLVER"],
    "cublas_cusolver" => ["cublas_cusolver"],
    "magma"    => ["MAGMA"],
    "jax"      => ["JAX (vmap)"],
    "cpu"      => ["CPU (multithreaded)"],
)

const MAX_WARPS_PER_SM = 48

const SHMEM_BYTES_PER_WAVEFRONT = 128.0


"""
    bench_times(op, impl) -> Dict{Int,Float64}

Read the benchmark timing table
benchmarking/benchmarks/bench_<op>/tables/<op>.csv and return
D -> time per matrix (seconds) for `impl`.
"""
function bench_times(op::AbstractString, impl::AbstractString)
    path = _rel(joinpath("..", "benchmarks", "bench_$(op)", "tables", "$(op).csv"))
    isfile(path) || error("benchmark table not found: $path")
    lines = readlines(path)
    header = String.(split(strip(lines[1]), ","))

    candidates = get(BENCH_COL, impl, nothing)
    candidates === nothing && error("no benchmark column mapping for impl '$impl'")
    cidx = nothing
    for c in candidates
        cidx = findfirst(==(c), header)
        cidx === nothing || break
    end
    cidx === nothing && error("none of $(candidates) found in $path (have: $(header))")

    didx = findfirst(==("D"), header)
    out = Dict{Int,Float64}()
    for li in 2:length(lines)
        isempty(strip(lines[li])) && continue
        f = split(strip(lines[li]), ",")
        out[parse(Int, strip(f[didx]))] = parse(Float64, strip(f[cidx]))
    end
    return out
end

const STALL_KEYS = [
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"  => "long_scoreboard (memory dependency / latency-bound)",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio" => "short_scoreboard (MIO dependency)",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"     => "mio_throttle (LSU/shared-mem throughput-bound)",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"      => "lg_throttle (global LSU queue)",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"          => "barrier (CTA sync)",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"             => "wait (fixed-latency exec dependency)",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"=> "math_pipe_throttle (compute-bound)",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"     => "not_selected (scheduler oversubscribed)",
]

function dominant_stall(rows)
    best = ("none", -1.0)
    for (k, label) in STALL_KEYS
        v = _first_metric(rows, k)
        if v isa Number && v > best[2]
            best = (label, v)
        end
    end
    return best
end

# The four launch__occupancy_limit_* block limits.
function occupancy_limits(rows)
    return (
        blocks     = _first_metric(rows, "launch__occupancy_limit_blocks"),
        registers  = _first_metric(rows, "launch__occupancy_limit_registers"),
        shared_mem = _first_metric(rows, "launch__occupancy_limit_shared_mem"),
        warps      = _first_metric(rows, "launch__occupancy_limit_warps"),
    )
end

# Occupancy limiter: smallest of the four limits; ties reported joined.
function occupancy_limiter(lims; tol::Float64=1e-6)
    valid = [(k, v) for (k, v) in pairs(lims) if v isa Number]
    isempty(valid) && return ("unknown", missing)
    minval = minimum(v for (_, v) in valid)
    tied = sort([String(k) for (k, v) in valid if v <= minval + tol])
    return (join(tied, "+"), minval)
end

# Theoretical occupancy
#   theoretical_occ = min_block_limit * warps_per_block / MAX_WARPS_PER_SM
#   warps_per_block = block_size / 32.
function theoretical_occupancy_pct(rows, lims)
    bs = _first_metric(rows, "launch__block_size")
    (bs isa Number) || return missing
    valid = [v for (_, v) in pairs(lims) if v isa Number]
    isempty(valid) && return missing
    min_blocks = minimum(valid)
    warps_per_block = bs / 32
    return 100.0 * min_blocks * warps_per_block / MAX_WARPS_PER_SM
end

is_baseline(impl::AbstractString) = impl != "ours"

# Build a result row with every diagnostic field set to `missing`.
# Baselines fill only the roofline-coordinate fields, keeping the
# NamedTuple shape identical to the `ours` path
function _blank_row(; D, t_pbe, fl_pbe, achieved_flops, dram_pbe, bytes_src,
                      analytic_dram_pbe, dram_AI, N, analytic_dram_tot,
                      meas_dram, dram_discrepancy_pct, n_kernels)
    return (
        D = D, time_per_batch_elem_s = t_pbe, flops_per_batch_elem = fl_pbe,
        achieved_flops = achieved_flops, dram_bytes_per_batch_elem = dram_pbe,
        dram_bytes_source = bytes_src, analytic_dram_bytes_per_batch_elem = analytic_dram_pbe,
        dram_AI = dram_AI, shmem_bytes_per_batch_elem = missing, shmem_AI = missing,
        batch_size = N, analytic_dram_tot = analytic_dram_tot,
        measured_dram_tot = meas_dram, dram_discrepancy_pct = dram_discrepancy_pct,
        dram_throughput_pct = missing, sm_throughput_pct = missing,
        l2_throughput_pct = missing, occupancy_pct = missing,
        theoretical_occupancy_pct = missing, occ_limiter = "",
        occ_limit_blocks = missing, occ_limit_registers = missing,
        occ_limit_shared_mem = missing, occ_limit_warps = missing,
        occ_limit_blocks_max = missing, shared_mem_per_block_B = missing,
        regs_per_thread = missing, l1_hit_rate_pct = missing,
        l2_hit_rate_pct = missing, lsu_util_pct = missing,
        alu_util_pct = missing, fma_util_pct = missing, adu_util_pct = missing,
        xu_util_pct = missing, bank_conflict_pct = missing,
        bank_max_nway = missing, bank_n_conflicting_insts = missing,
        dominant_instruction = missing, dominant_instruction_pct = missing,
        top_instructions = missing, dominant_stall = "",
        dominant_stall_val = missing, n_kernels = n_kernels,
    )
end

"""
    analyse(op, impl, Ds; profile_dir="profile_results", write_csv=true)

Compute roofline-ready quantities for one (operation, implementation)
across the dimensions `Ds`, write a tidy CSV, print a summary.

Two modes:
  * impl == "ours"  -- reads the full --set full diagnostic profile
    (<op>_ours_D<d>.csv from run_profiles.sh) and the SASS CSV, and
    produces the complete diagnostic suite plus roofline coordinates.
  * a baseline      -- contributes only a roofline point: benchmark time
    + DRAM bytes (analytic, or measured via measure_baseline for a
    multi-kernel non-fused baseline). Diagnostic fields are `missing`.

Roofline coordinates use ANALYTIC flops always. DRAM bytes are analytic
for single-pass implementations and MEASURED for multi-kernel baselines.
"""
function analyse(op::AbstractString, impl::AbstractString, Ds;
                 profile_dir::AbstractString="profile_results",
                 write_csv::Bool=true, force::Bool=false)

    times = bench_times(op, impl)
    results = NamedTuple[]
    dir_base = _rel(op)

    for D in Ds
        haskey(times, D) || error("no benchmark time for D=$D in table")
        row = is_baseline(impl) ?
              _analyse_baseline(op, impl, D, times[D], dir_base, profile_dir; force=force) :
              _analyse_ours(op, impl, D, times[D], dir_base, profile_dir)
        push!(results, row)
    end

    _print_table(op, impl, results)

    if write_csv
        outpath = joinpath(dir_base, profile_dir, "analysis_$(op)_$(impl).csv")
        _write_tidy(outpath, op, impl, results)
        println("\nwrote $outpath")
    end

    return results
end

# Baseline: roofline point only
# A benchmark time and a byte count.
# No full profile is read, setting all diagnostic fields to `missing`
# Whether the baseline is multi-kernel is determined empirically by
# discovery (cheap nsys pass, cached), not hard-coded. A multi-kernel
# baseline gets 'measured' bytes, a single-kernel one uses analytic bytes.
function _analyse_baseline(op, impl, D, t_pbe, dir_base, profile_dir; force::Bool=false)
    # pbe: per batch element, i.e. per N
    fl_pbe = flops_per_batch_elem(op, D)
    analytic_dram_pbe = dram_bytes_per_batch_elem(op, D)
    N = batch_size(op, D)

    # Determine whether this baseline is multi-kernel. If a counting
    # driver is registered (COUNT_DRIVERS), discovery decides empirically
    # (cached unless force=true). If no driver is registered, the baseline
    # is assumed to be single-kernel.
    # The registry is the list of baselines worth checking.
    # An unregistered baseline defaults to analytic bytes.
    # This default is taken for matmmul, matadd, etc.
    disc = _ensure_discovery(op, impl, D, dir_base, profile_dir; force=force)

    # bytes: measured for a multi-kernel non-fused baseline, else analytic
    if disc.multi_kernel
        mb = measure_baseline(op, impl, D; force=force,
                                      profile_dir=joinpath(dir_base, profile_dir))
        dram_pbe    = mb.measured_bytes_per_batch_elem
        bytes_src  = "measured"
        n_kernels  = mb.n_kernels_captured
        # total_measured_dram is exactly ONE steady-state call (marker-
        # delimited); that is the per-N total the cross-check wants.
        meas_dram  = mb.total_measured_dram
    else
        dram_pbe    = analytic_dram_pbe
        bytes_src  = "analytic"
        meas_dram  = missing
        n_kernels  = 1
    end

    achieved_flops = fl_pbe / t_pbe
    dram_AI        = fl_pbe / dram_pbe
    analytic_dram_tot = analytic_dram_pbe * N
    dram_discrepancy_pct = meas_dram isa Number ?
        100.0 * (meas_dram - analytic_dram_tot) / analytic_dram_tot : missing

    return _blank_row(; D=D, t_pbe=t_pbe, fl_pbe=fl_pbe,
        achieved_flops=achieved_flops, dram_pbe=dram_pbe, bytes_src=bytes_src,
        analytic_dram_pbe=analytic_dram_pbe, dram_AI=dram_AI, N=N,
        analytic_dram_tot=analytic_dram_tot, meas_dram=meas_dram,
        dram_discrepancy_pct=dram_discrepancy_pct, n_kernels=n_kernels)
end

# Ensure kernel-discovery has run for (op, impl, D) and return its
# scalars: multi_kernel (load-bearing -- does this baseline need a
# measured-bytes run?) and kernels_per_call (a non-binding cross-check).
#
# discover_kernels writes <op>_<impl>_D<d>_discovery.csv; if present it
# is reused. Otherwise discovery runs now -- an nsys pass, cheap relative
# to the ncu measurement, and cached.
function _ensure_discovery(op, impl, D, dir_base, profile_dir; force::Bool=false)
    # D-tagged: kernels-per-call is not D-independent for every baseline
    scalar_csv = joinpath(dir_base, profile_dir, "$(op)_$(impl)_D$(D)_discovery.csv")
    if isfile(scalar_csv) && !force
        return _read_discovery_cache(scalar_csv)   # from measure_baseline.jl
    end

    # No cached discovery. If a counting driver is registered for this
    # baseline, run discovery empirically.
    # If not, the absence of a driver is itself the declaration
    # this baseline is single-kernel (like matmul)
    # This is to avoid redundant checking if we already know that it's single-kernel
    if !haskey(COUNT_DRIVERS, (op, impl))
        @info "no counting driver for ($op,$impl) -- assuming single-kernel, " *
              "using analytic DRAM bytes. If this baseline is non-fused, add " *
              "a driver to COUNT_DRIVERS in discover_kernels.jl."
        return (kernels_per_call = 1, multi_kernel = false)
    end

    println("  no discovery for ($op,$impl) -- running discover_kernels...")
    d = discover_kernels(op, impl; D=D,
                         profile_dir=joinpath(dir_base, profile_dir))
    return (kernels_per_call = d.kernels_per_call,
            multi_kernel     = d.multi_kernel)
end

# `ours`: reads the full profile CSV and
# the SASS CSV, computes the complete diagnostic report
function _analyse_ours(op, impl, D, t_pbe, dir_base, profile_dir)
    csv = joinpath(dir_base, profile_dir, "$(op)_$(impl)_D$(D).csv")
    isfile(csv) || error("missing profile CSV: $csv")
    rows = parse_ncu_csv(csv)  # full profile CSV

    # analytic (per batch element)
    fl_pbe            = flops_per_batch_elem(op, D)
    analytic_dram_pbe = dram_bytes_per_batch_elem(op, D)

    # `ours` is single-pass: analytic DRAM bytes are the true minimum.
    dram_pbe   = analytic_dram_pbe
    bytes_src = "analytic"

    # roofline coordinates (N cancels)
    achieved_flops = fl_pbe / t_pbe
    dram_AI        = fl_pbe / dram_pbe

    # measured DRAM total + cross-check vs analytic
    rd = _sum_metric(rows, "dram__bytes_read.sum")
    wr = _sum_metric(rows, "dram__bytes_write.sum")
    meas_dram = (rd isa Number && wr isa Number) ? rd + wr : missing
    N = batch_size(op, D)
    analytic_dram_tot = analytic_dram_pbe * N
    dram_discrepancy_pct = meas_dram isa Number ?
        100.0 * (meas_dram - analytic_dram_tot) / analytic_dram_tot : missing

    # throughput / occupancy
    dram_throughput_pct = _first_metric(rows, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
    sm_tput  = _first_metric(rows, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    l2_tput  = _first_metric(rows, "lts__throughput.avg.pct_of_peak_sustained_elapsed")
    occ_pct  = _first_metric(rows, "sm__warps_active.avg.pct_of_peak_sustained_active")
    smem_pb  = _first_metric(rows, "launch__shared_mem_per_block")
    regs_pt  = _first_metric(rows, "launch__registers_per_thread")
    n_kernels = length(rows)

    lims = occupancy_limits(rows)
    lim_name, lim_blocks = occupancy_limiter(lims)
    theo_occ = theoretical_occupancy_pct(rows, lims)

    # cache hit rates
    l1_hit = _first_metric(rows, "l1tex__t_sector_hit_rate.pct")
    l2_hit = _first_metric(rows, "lts__t_sector_hit_rate.pct")

    # pipe utilisations (% of peak instruction issue)
    lsu_util = _first_metric(rows, "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active")
    alu_util = _first_metric(rows, "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active")
    fma_util = _first_metric(rows, "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active")
    adu_util = _first_metric(rows, "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active")
    xu_util  = _first_metric(rows, "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active")

    # shared-memory bank conflicts + instruction mix (SASS CSV)
    sass_csv = joinpath(dir_base, profile_dir, "$(op)_$(impl)_D$(D)_sass.csv")
    bank_conflict_pct = missing; bank_max_nway = missing
    bank_n_conflicting = missing; dominant_inst = missing
    dominant_inst_pct = missing; top_insts = missing
    shmem_bytes_per_batch_elem = missing; shmem_AI = missing
    if isfile(sass_csv)
        bc = bank_conflicts(sass_csv)
        bank_conflict_pct  = bc.conflict_pct
        bank_max_nway      = bc.max_nway
        bank_n_conflicting = length(bc.conflicting)
        mix = instruction_mix(sass_csv)
        dominant_inst     = mix.dominant
        dominant_inst_pct = mix.dominant_pct
        top_insts         = instruction_mix_summary(mix)

        # shared-memory roofline x-axis. bc.total_shared is the sum of
        # `L1 Wavefronts Shared` over every shared LDS/STS instruction
        # See shmem_roofline_derivation.md.
        shmem_bytes_per_batch_elem = bc.total_shared * SHMEM_BYTES_PER_WAVEFRONT / N
        shmem_AI       = shmem_bytes_per_batch_elem > 0 ? fl_pbe / shmem_bytes_per_batch_elem : missing
    else
        @warn "SASS CSV not found, bank-conflict / instruction-mix / " *
              "shared-memory roofline fields left blank: $sass_csv"
    end

    stall_name, stall_val = dominant_stall(rows)

    return (
        D                 = D,
        time_per_batch_elem_s = t_pbe,
        flops_per_batch_elem  = fl_pbe,
        achieved_flops    = achieved_flops,
        dram_bytes_per_batch_elem     = dram_pbe,
        dram_bytes_source = bytes_src,
        analytic_dram_bytes_per_batch_elem = analytic_dram_pbe,
        dram_AI           = dram_AI,
        shmem_bytes_per_batch_elem    = shmem_bytes_per_batch_elem,
        shmem_AI          = shmem_AI,
        batch_size        = N,
        analytic_dram_tot = analytic_dram_tot,
        measured_dram_tot = meas_dram,
        dram_discrepancy_pct = dram_discrepancy_pct,
        dram_throughput_pct  = dram_throughput_pct,
        sm_throughput_pct = sm_tput,
        l2_throughput_pct = l2_tput,
        occupancy_pct     = occ_pct,
        theoretical_occupancy_pct = theo_occ,
        occ_limiter       = lim_name,
        occ_limit_blocks  = lim_blocks,
        occ_limit_registers  = lims.registers,
        occ_limit_shared_mem = lims.shared_mem,
        occ_limit_warps   = lims.warps,
        occ_limit_blocks_max = lims.blocks,
        shared_mem_per_block_B = smem_pb,
        regs_per_thread   = regs_pt,
        l1_hit_rate_pct   = l1_hit,
        l2_hit_rate_pct   = l2_hit,
        lsu_util_pct      = lsu_util,
        alu_util_pct      = alu_util,
        fma_util_pct      = fma_util,
        adu_util_pct      = adu_util,
        xu_util_pct       = xu_util,
        bank_conflict_pct = bank_conflict_pct,
        bank_max_nway     = bank_max_nway,
        bank_n_conflicting_insts = bank_n_conflicting,
        dominant_instruction     = dominant_inst,
        dominant_instruction_pct = dominant_inst_pct,
        top_instructions  = top_insts,
        dominant_stall    = stall_name,
        dominant_stall_val= stall_val,
        n_kernels         = n_kernels,
    )
end

# dominant pipe: the highest-utilised instruction pipe for this kernel,
# formatted "NAME pct".
# Max over all five collected pipes:
# - LSU (load/store, shmem pressure)
# - ALU
# - FMA (fused multiply-add)
# - ADU (address generation),
# XU (special-function unit: sqrt, rcp, sin/cos/exp
# Whichever is highest is the pipe the kernel actually leans on.
# Full five are in the CSV, the table
# shows only the dominant one
function _dom_pipe(r)
    pipes = (("LSU", r.lsu_util_pct), ("ALU", r.alu_util_pct),
             ("FMA", r.fma_util_pct), ("ADU", r.adu_util_pct),
             ("XU",  r.xu_util_pct))
    avail = [(n, v) for (n, v) in pipes if v isa Number]
    isempty(avail) && return "-"
    name, val = avail[argmax(last.(avail))]
    return "$name $(round(val, digits=1))"
end

function _print_table(op, impl, results)
    println("\n=== $op / $impl ===")
    println(rpad("D",4), rpad("t/batch_el(s)",14), rpad("GFLOP/s",10),
            rpad("AI",9), rpad("DRAM%",7), rpad("dom_pipe",11), rpad("occ%",7),
            rpad("bconf%",8), rpad("dom_inst",10), "dominant stall")
    println("-"^115)
    for r in results
        gf  = isnan(r.achieved_flops) ? "-" : string(round(r.achieved_flops/1e9, digits=1))
        # AI marked with '*' when the roofline used MEASURED dram bytes
        # (multi-kernel baseline) rather than the analytic single-pass count
        ai  = string(round(r.dram_AI, digits=3)) *
              (r.dram_bytes_source == "measured" ? "*" : "")
        dr  = r.dram_throughput_pct isa Number ? string(round(r.dram_throughput_pct,digits=1)) : "-"
        dp  = _dom_pipe(r)
        oc  = r.occupancy_pct       isa Number ? string(round(r.occupancy_pct,digits=1))       : "-"
        bc  = r.bank_conflict_pct   isa Number ?
              string(round(r.bank_conflict_pct, digits=2)) : "-"
        di  = r.dominant_instruction isa AbstractString ? r.dominant_instruction : "-"
        ds  = (r.dominant_stall isa AbstractString && !isempty(r.dominant_stall)) ?
              r.dominant_stall : "-"
        println(rpad(r.D,4), rpad(round(r.time_per_batch_elem_s,sigdigits=4),14),
                rpad(gf,10), rpad(ai,9), rpad(dr,7), rpad(dp,11), rpad(oc,7),
                rpad(bc,8), rpad(di,10), ds)
    end
    any(r -> r.dram_bytes_source == "measured", results) &&
        println("(* AI from measured DRAM bytes -- multi-kernel non-fused baseline)")
end

function _write_tidy(path, op, impl, results)
    cols = ["operation","implementation","D","time_per_batch_elem_s","flops_per_batch_elem",
            "achieved_flops","dram_bytes_per_batch_elem","dram_bytes_source",
            "analytic_dram_bytes_per_batch_elem","dram_AI","shmem_bytes_per_batch_elem","shmem_AI",
            "batch_size","analytic_dram_tot",
            "measured_dram_tot","dram_discrepancy_pct","dram_throughput_pct",
            "sm_throughput_pct","l2_throughput_pct","occupancy_pct",
            "theoretical_occupancy_pct","occ_limiter","occ_limit_blocks",
            "occ_limit_registers","occ_limit_shared_mem","occ_limit_warps",
            "occ_limit_blocks_max","shared_mem_per_block_B","regs_per_thread",
            "l1_hit_rate_pct","l2_hit_rate_pct","lsu_util_pct","alu_util_pct",
            "fma_util_pct","adu_util_pct","xu_util_pct","bank_conflict_pct",
            "bank_max_nway","bank_n_conflicting_insts","dominant_instruction",
            "dominant_instruction_pct","top_instructions","dominant_stall",
            "dominant_stall_val","n_kernels"]
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in results
            println(io, join((
                op, impl, r.D, r.time_per_batch_elem_s, r.flops_per_batch_elem,
                r.achieved_flops, r.dram_bytes_per_batch_elem, r.dram_bytes_source,
                r.analytic_dram_bytes_per_batch_elem, r.dram_AI,
                _csv(r.shmem_bytes_per_batch_elem), _csv(r.shmem_AI), r.batch_size,
                r.analytic_dram_tot, _csv(r.measured_dram_tot),
                _csv(r.dram_discrepancy_pct), _csv(r.dram_throughput_pct),
                _csv(r.sm_throughput_pct), _csv(r.l2_throughput_pct),
                _csv(r.occupancy_pct), _csv(r.theoretical_occupancy_pct),
                r.occ_limiter, _csv(r.occ_limit_blocks),
                _csv(r.occ_limit_registers), _csv(r.occ_limit_shared_mem),
                _csv(r.occ_limit_warps), _csv(r.occ_limit_blocks_max),
                _csv(r.shared_mem_per_block_B), _csv(r.regs_per_thread),
                _csv(r.l1_hit_rate_pct), _csv(r.l2_hit_rate_pct),
                _csv(r.lsu_util_pct), _csv(r.alu_util_pct), _csv(r.fma_util_pct),
                _csv(r.adu_util_pct), _csv(r.xu_util_pct),
                _csv(r.bank_conflict_pct), _csv(r.bank_max_nway),
                _csv(r.bank_n_conflicting_insts),
                _csv(r.dominant_instruction), _csv(r.dominant_instruction_pct),
                _csv(r.top_instructions),
                r.dominant_stall, _csv(r.dominant_stall_val), r.n_kernels,
            ), ","))
        end
    end
end

_csv(x) = x === missing ? "" : string(x)
