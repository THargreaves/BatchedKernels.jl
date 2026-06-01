#!/usr/bin/env julia
# ======================================================================
# decide_nthreads.jl  --  STEP 3 of the nthreads tuning pipeline.
#
# Combines STEP 2's benchmarks into the deliverable schedule:
#
#   1. per (op, D): the winning nthreads = argmin(median_time).
#      Cells STEP 2 did not benchmark (below the tuning window) keep 256.
#   2. fit a TWO-SWITCH step rule per op: nthreads = 256 outside
#      [D_low, D_high) and nt_mid inside. Captures the two regimes seen
#      in the data: occupancy-collapse in a middle band (smaller block
#      wins), and work-amortisation at very large D (256 wins again).
#      A one-switch rule cannot represent that.
#   3. report the speedup of the chosen rule vs the fixed-256 baseline.
#
# Outputs: ../config/nthreads_schedule.csv  (canonical; read by all studies)
#          results/TUNING.md                (auto-generated summary)
# ======================================================================

using Printf
using Dates
using Statistics

const TUNE_DIR    = @__DIR__
const RESULTS_DIR = joinpath(TUNE_DIR, "results")
# the schedule is written to studies/config/ -- the single source of
# truth read by roofline, benchmarks, and any other study via
# config/Schedule.jl. tune_nthreads/ is studies/tune_nthreads/, so
# config/ is one level up.
const CONFIG_DIR  = abspath(joinpath(TUNE_DIR, "..", "config"))
const BENCH_CSV   = joinpath(RESULTS_DIR, "nthreads_benchmarks.csv")
const PRED_CSV    = joinpath(RESULTS_DIR, "occupancy_prediction.csv")

const OPS      = ["kalman", "sqrt_kalman", "gauss_likelihood", "qr_r"]
const NTHREADS = [64, 96, 128, 160, 192, 224, 256]

# Minimum measured speedup required to commit to switching block size.
# fit_step_rule's sum-minimisation can land boundaries in the noise region
# at low D (per-D speedups < 1.02 there are measurement jitter, not real
# wins -- but the high-D wins still pull the global sum-optimum down).
# After the fit, we ADVANCE each boundary to the smallest D where the rule's
# chosen block actually beats 256 by at least this fraction -- so the
# schedule only commits to a switch where it pays off measurably.
# 5% is well above the ~1-3% per-D noise floor observed in benchmarks.
const MIN_SPEEDUP_THRESHOLD = 1.05

# Per-D override threshold. After the rule is built and its per-D
# choices are emitted, any single (op, D) cell where the measured
# per-D argmin beats the rule's chosen block by >= this fraction is
# overridden to use the argmin instead. Targets cells where the single
# rule's nt_mid happens to be much worse at one D than another block
# size would be (e.g. sqrt_kalman D=20-21: rule says 256, nt=160 is
# ~22% faster). 15% is well above the 5% boundary-fit threshold so the
# override fires only on substantial wins, not noise.
const OVERRIDE_SPEEDUP_THRESHOLD = 1.15

# ----------------------------------------------------------------------
# Shared-memory feasibility -- mirrors benchmark_nthreads.jl (and the
# formulas in predict_occupancy.jl / each kernel's launch code). Needed
# here because the fitted step rule's "256 below the switch" branch is
# NOT valid at large D: kalman/sqrt_kalman cannot run at nthreads=256
# for D >= 29. Any schedule entry must be clamped to a feasible block.
# ----------------------------------------------------------------------
const SHMEM_CARVEOUT = 101376    # RTX 4090 opt-in max, bytes (conservative)

function _shmem_bytes(op::AbstractString, D::Int, nthreads::Int)
    n_mats_per_warp = 32 ÷ D
    n_warps         = nthreads ÷ 32
    dual_padding    = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems     = warp_shmem_size * n_warps
    if op == "gauss_likelihood"
        shmem_vec_elems = D * n_warps * n_mats_per_warp
        return sizeof(Float32) * (2 * shmem_elems + 2 * shmem_vec_elems)
    elseif op in ("kalman", "sqrt_kalman")
        pad_interval     = div(32, D & -D) * D
        shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval
        return sizeof(Float32) * (3 * shmem_elems + 4 * shmem_size_fixed)
    elseif op in ("qr_r",)
        return sizeof(Float32) * 2 * shmem_elems
    else
        error("no shared-memory formula for op '$op'")
    end
end

_shmem_feasible(op, D, nt) = _shmem_bytes(op, D, nt) <= SHMEM_CARVEOUT

# largest candidate nthreads <= `desired` that fits (op,D) in shared
# memory. Used to clamp the step rule's choice at large D, where the
# rule's nominal value (often 256) may be infeasible.
function _feasible_nthreads(op, D, desired::Int)
    cands = sort([nt for nt in NTHREADS
                  if nt <= desired && _shmem_feasible(op, D, nt)];
                 rev = true)
    isempty(cands) && return minimum(NTHREADS)   # nothing fits: smallest
    return first(cands)
end

# ---- read STEP 2 benchmarks --------------------------------------------
# returns time[(op,D,nt)] and a set of benchmarked (op,D), and N[(op,D)]
function read_benchmarks()
    isfile(BENCH_CSV) || error("$BENCH_CSV not found -- run STEP 2 first.")
    lines  = readlines(BENCH_CSV)
    header = split(strip(lines[1]), ",")
    ci(c)  = findfirst(==(c), header)
    iop,iD,int,itm,iN,ibm =
        ci("op"),ci("D"),ci("nthreads"),ci("median_time_s"),
        ci("batch_size"),ci("benchmarked")
    time = Dict{Tuple{String,Int,Int},Float64}()
    Nof  = Dict{Tuple{String,Int},Int}()
    benched = Set{Tuple{String,Int}}()
    for li in 2:length(lines)
        f = split(strip(lines[li]), ",")
        length(f) < length(header) && continue
        op = String(f[iop]); D = parse(Int,f[iD]); nt = parse(Int,f[int])
        Nof[(op,D)] = parse(Int, f[iN])
        if strip(f[ibm]) == "true" && !isempty(strip(f[itm]))
            time[(op,D,nt)] = parse(Float64, f[itm])
            push!(benched, (op,D))
        end
    end
    return time, benched, Nof
end

# ---- read STEP 1 predicted occupancy (for the TUNING.md comparison) ----
function read_predicted_occ()
    occ = Dict{Tuple{String,Int,Int},Float64}()
    isfile(PRED_CSV) || return occ
    lines  = readlines(PRED_CSV)
    header = split(strip(lines[1]), ",")
    ci(c)  = findfirst(==(c), header)
    iop,iD,int,io = ci("op"),ci("D"),ci("nthreads"),
                    ci("predicted_occupancy_pct")
    for li in 2:length(lines)
        f = split(strip(lines[li]), ",")
        length(f) < length(header) && continue
        occ[(String(f[iop]),parse(Int,f[iD]),parse(Int,f[int]))] =
            parse(Float64, f[io])
    end
    return occ
end

# per-D winner: argmin median time among benchmarked nthreads
function per_d_winner(time, op, D)
    cands = [(nt, time[(op,D,nt)]) for nt in NTHREADS
             if haskey(time,(op,D,nt))]
    isempty(cands) && return (nthreads=256, time=NaN)
    nt, t = cands[argmin(last.(cands))]
    return (nthreads=nt, time=t)
end

# fit a TWO-SWITCH step rule: pick (D_low, D_high, nt_mid) minimising
# mean time per scored D. The rule has THREE regions:
#     D <  D_low                 -> 256                  (small D)
#     D_low  <= D < D_high       -> nt_mid               (middle band)
#     D_high <= D                -> 256                  (very large D)
#
# Motivation: kalman/sqrt_kalman show TWO transitions, not one. In a
# middle D band the per-block shared memory crushes 256-thread occupancy
# (smaller block wins by 10-60%); but at very large D every block fits
# only once per SM regardless of block size, and 256 wins again because
# the per-block work amortises the launch/setup overhead better. A
# single-switch rule cannot represent that -- it picks one region and
# misuses the other.
#
# `D_low == D_high` collapses to "never switch" (rule = 256 everywhere);
# `D_high == max(D)+1` collapses to a single-switch rule that runs all
# the way to the end (back-compat with the previous behaviour).
#
# Infeasibility handling (kalman D>=29 at 256, etc.) is identical to the
# single-switch fit: skip D's the candidate rule cannot score, average
# over the rest so rules covering different D-sets stay comparable.
function fit_step_rule(time, op, benched_Ds)
    isempty(benched_Ds) && return (D_low=typemax(Int), D_high=typemax(Int),
                                   nt_mid=256)
    Dmin, Dmax = minimum(benched_Ds), maximum(benched_Ds)
    best = (D_low=typemax(Int), D_high=typemax(Int), nt_mid=256, cost=Inf)
    for nt_mid in NTHREADS,
        D_low  in Dmin:(Dmax+1),
        D_high in D_low:(Dmax+1)
        total  = 0.0
        scored = 0
        for D in benched_Ds
            nt = (D < D_low || D >= D_high) ? 256 : nt_mid
            haskey(time,(op,D,nt)) || continue
            total  += time[(op,D,nt)]
            scored += 1
        end
        scored == 0 && continue
        cost = total / scored
        if cost < best.cost
            best = (D_low=D_low, D_high=D_high, nt_mid=nt_mid, cost=cost)
        end
    end
    return (D_low=best.D_low, D_high=best.D_high, nt_mid=best.nt_mid)
end

# helper: apply a two-switch rule at a single D
_rule_nt(rule, D::Integer) =
    (D < rule.D_low || D >= rule.D_high) ? 256 : rule.nt_mid

# Post-filter both boundaries independently. Each switch must be backed
# by a >= MIN_SPEEDUP_THRESHOLD measured win at its boundary D, else it
# is moved past the noisy region.
#
#   * `D_low`  : the rule's chosen nt_mid must beat 256 at D=D_low by
#                >= threshold; if not, advance D_low until it does.
#   * `D_high` : at D=D_high the rule switches BACK to 256, so 256 must
#                beat nt_mid at D=D_high by >= threshold; if not, advance
#                D_high until it does, or out of range (no upper switch).
#
# As before: filter only advances boundaries (or collapses the rule);
# it never makes the rule more aggressive than the fit suggested.
function apply_min_speedup_filter(rule, time, op, benched_Ds,
                                  threshold::Real)
    isempty(benched_Ds) && return rule
    Dmin, Dmax = minimum(benched_Ds), maximum(benched_Ds)

    # ---- lower boundary: nt_mid must beat 256 by >= threshold at D_low
    new_low = rule.D_low
    if new_low <= Dmax
        while new_low <= Dmax
            t256 = get(time, (op, new_low, 256),         nothing)
            tmid = get(time, (op, new_low, rule.nt_mid), nothing)
            if t256 !== nothing && tmid !== nothing && tmid > 0 &&
               t256 / tmid >= threshold
                break
            end
            new_low += 1
        end
        if new_low != rule.D_low
            if new_low > Dmax
                println("  $op: lower switch (D_low=$(rule.D_low)) " *
                        "fails $(threshold)x test; rule collapses to 256")
            else
                println("  $op: D_low advanced $(rule.D_low) -> $new_low " *
                        "(first D where nt_mid=$(rule.nt_mid) " *
                        "beats 256 by >= $(threshold)x)")
            end
        end
    end

    # ---- upper boundary: 256 must beat nt_mid by >= threshold at D_high
    # if D_high > Dmax already, there's no upper switch to verify
    new_high = rule.D_high
    if rule.D_high <= Dmax
        while new_high <= Dmax
            t256 = get(time, (op, new_high, 256),         nothing)
            tmid = get(time, (op, new_high, rule.nt_mid), nothing)
            # at D_high the rule switches BACK to 256, so 256 must win
            if t256 !== nothing && tmid !== nothing && t256 > 0 &&
               tmid / t256 >= threshold
                break
            end
            new_high += 1
        end
        if new_high != rule.D_high
            if new_high > Dmax
                println("  $op: upper switch back to 256 (D_high=" *
                        "$(rule.D_high)) fails $(threshold)x test; " *
                        "rule stays at nt_mid past benchmark window")
                new_high = Dmax + 1   # effectively "no upper switch"
            else
                println("  $op: D_high advanced $(rule.D_high) -> $new_high " *
                        "(first D where 256 beats nt_mid=$(rule.nt_mid) " *
                        "by >= $(threshold)x)")
            end
        end
    end

    # if the lower-boundary filter pushed D_low past the upper, the
    # whole middle band has vanished -- collapse to "never switch"
    if new_low >= new_high
        return (D_low=typemax(Int), D_high=typemax(Int), nt_mid=256)
    end
    return (D_low=new_low, D_high=new_high, nt_mid=rule.nt_mid)
end

# ----------------------------------------------------------------------
# render a markdown table with PADDED columns so the raw .md is readable
# (GitHub etc. render either way; padding is for humans reading the file).
#   header :: Vector{String}    -- column titles
#   rows   :: Vector{Vector{String}}
# ----------------------------------------------------------------------
function _mdtable(io, header::Vector{String}, rows::Vector{Vector{String}})
    ncol  = length(header)
    width = [length(header[c]) for c in 1:ncol]
    for r in rows, c in 1:ncol
        width[c] = max(width[c], length(r[c]))
    end
    pad(s, w) = s * repeat(" ", w - length(s))
    line(cells) = "| " * join((pad(cells[c], width[c]) for c in 1:ncol),
                               " | ") * " |"
    println(io, line(header))
    println(io, "| " * join((repeat("-", width[c]) for c in 1:ncol),
                            " | ") * " |")
    for r in rows
        println(io, line(r))
    end
end

# human-readable string for a two-switch rule.
function _format_rule(rule, allDs)
    if rule.D_low == typemax(Int)
        return "256 for all D"
    end
    upper = if isempty(allDs) || rule.D_high > maximum(allDs)
        ""    # rule never switches back -- nt_mid runs to end of range
    else
        ", then 256 for D>=$(rule.D_high)"
    end
    return "256 for D<$(rule.D_low), $(rule.nt_mid) for $(rule.D_low)<=D<$(rule.D_high)$upper"
end

function main()
    time, benched, Nof = read_benchmarks()
    occ = read_predicted_occ()

    schedule = Dict{Tuple{String,Int},Int}()   # (op,D) -> chosen nthreads
    rules    = Dict{String,NamedTuple}()
    allDs    = sort(unique(d for (_,d) in keys(Nof)))

    for op in OPS
        bDs = sort([d for (o,d) in benched if o == op])
        rule = fit_step_rule(time, op, bDs)
        rule = apply_min_speedup_filter(rule, time, op, bDs,
                                        MIN_SPEEDUP_THRESHOLD)
        rules[op] = rule
        for D in allDs
            haskey(Nof,(op,D)) || continue
            nominal = _rule_nt(rule, D)

            # Per-D override: if the measured argmin at this D beats the
            # rule's nominal choice by >= OVERRIDE_SPEEDUP_THRESHOLD,
            # override THIS D only (does NOT change the rule's boundaries
            # for other D's). Targets cells where a single nt_mid is much
            # worse at one D than another candidate would be.
            t_nom = get(time, (op, D, nominal), nothing)
            w = per_d_winner(time, op, D)
            if t_nom !== nothing && !isnan(w.time) && w.nthreads != nominal &&
               w.time > 0 && t_nom / w.time >= OVERRIDE_SPEEDUP_THRESHOLD
                println("  $op D=$D: rule gave $nominal, " *
                        "overridden to $(w.nthreads) " *
                        "(argmin beats rule by " *
                        "$(round((t_nom/w.time - 1)*100, digits=1))%)")
                nominal = w.nthreads
            end

            # the rule's nominal block size may exceed shared memory at
            # large D (kalman/sqrt_kalman, D>=29 at 256). Clamp down to
            # the largest feasible candidate so the schedule never emits
            # an entry that cannot launch.
            chosen  = _feasible_nthreads(op, D, nominal)
            if chosen != nominal
                println("  $op D=$D: clamped $nominal -> $chosen " *
                        "(shared-memory limit)")
            end
            schedule[(op,D)] = chosen
        end
    end

    # ---- nthreads_schedule.csv : the deliverable --------------------
    # Written to studies/config/ -- the canonical location every other
    # study reads via config/Schedule.jl. Single copy, no drift.
    mkpath(CONFIG_DIR)
    sched_csv = joinpath(CONFIG_DIR, "nthreads_schedule.csv")
    open(sched_csv, "w") do io
        println(io, "op,D,best_nthreads,per_d_argmin_nthreads," *
                    "per_d_argmin_time_s,rule_time_s,speedup_vs_256")
        for op in OPS, D in allDs
            haskey(schedule,(op,D)) || continue
            chosen = schedule[(op,D)]
            w  = per_d_winner(time, op, D)
            t256  = get(time,(op,D,256), NaN)
            trule = get(time,(op,D,chosen), t256)
            spd   = (isnan(t256)||isnan(trule)||trule==0) ? "" :
                    string(round(t256/trule, digits=4))
            println(io, join((op, D, chosen, w.nthreads,
                isnan(w.time) ? "" : round(w.time,sigdigits=6),
                isnan(trule)  ? "" : round(trule, sigdigits=6),
                spd), ","))
        end
    end
    println("wrote $sched_csv")

    # ---- TUNING.md : auto-generated methodology + results -----------
    md = joinpath(RESULTS_DIR, "TUNING.md")
    open(md, "w") do io
        println(io, "# nthreads tuning -- results\n")
        println(io, "_Generated $(now()) by decide_nthreads.jl._\n")
        println(io, "## Method\n")
        println(io, "Block size (`nthreads`) is tuned for the five ",
            "occupancy-bound operations: `kalman`, `sqrt_kalman`, ",
            "`gauss_likelihood`, `QR R`, `QR Q`.\n")
        println(io, "1. **Predict** (`predict_occupancy.jl`): occupancy ",
            "per (op,D,nthreads) from the kernels' analytic shared-memory ",
            "formula + registers/thread from the roofline profiles. ",
            "Locates the block-size threshold to within ~1 D.")
        println(io, "2. **Benchmark** (`benchmark_nthreads.jl`): median ",
            "kernel time for all nthreads in ",
            "{$(join(NTHREADS, ", "))}, over a D window around the ",
            "predicted threshold.")
        println(io, "3. **Decide** (this script): per-(op,D) argmin, then ",
            "a TWO-SWITCH step rule fitted per op: `nthreads = 256` ",
            "outside `[D_low, D_high)`, and `nt_mid` inside that band. ",
            "(Two switches because the data shows two regimes: in a ",
            "middle D band, occupancy collapse at 256 favours a smaller ",
            "block; at very large D, per-block work amortises launch ",
            "overhead and 256 wins again.) Each boundary is fitted by ",
            "summed-time minimisation, then post-filtered: each switch ",
            "must show a measured speedup of at least ",
            "$(round((MIN_SPEEDUP_THRESHOLD-1)*100; digits=1))% at its ",
            "boundary D, else the boundary is advanced past the noisy ",
            "region (or the rule collapses to plain 256). Finally, a ",
            "per-D override: any single (op, D) cell where the measured ",
            "argmin beats the rule's choice by >= ",
            "$(round((OVERRIDE_SPEEDUP_THRESHOLD-1)*100; digits=1))% is ",
            "set to the argmin -- handles cells the single-`nt_mid` ",
            "rule cannot capture well.\\n")
        println(io, "## Chosen schedule\n")
        sched_rows = Vector{String}[]
        for op in OPS
            r = rules[op]
            rule_s = _format_rule(r, allDs)
            # collect per-D overrides: cells where the schedule's chosen
            # nthreads differs from the rule's nominal choice. (Includes
            # both speedup-driven overrides and feasibility clamps.)
            ovs = Tuple{Int,Int,Int}[]
            for D in allDs
                haskey(schedule,(op,D)) || continue
                nom = _rule_nt(r, D)
                got = schedule[(op,D)]
                got == nom || push!(ovs, (D, nom, got))
            end
            ov_str = isempty(ovs) ? "(none)" :
                join(["D=$D: $nom→$got" for (D,nom,got) in ovs], ", ")
            t256  = get(time,(op,16,256), NaN)
            tr    = get(time,(op,16,get(schedule,(op,16),256)), NaN)
            spd   = (isnan(t256)||isnan(tr)||tr==0) ? "n/a" :
                    string(round(t256/tr, digits=3)) * "x"
            push!(sched_rows, ["`$op`", rule_s, ov_str, spd])
        end
        _mdtable(io, ["op", "rule", "per-D overrides",
                      "speedup at D=16 vs 256"], sched_rows)
        println(io)
        println(io, "## Per-(op,D) detail\n")
        for op in OPS
            println(io, "### `$op`\n")
            detail_rows = Vector{String}[]
            for D in allDs
                haskey(schedule,(op,D)) || continue
                w    = per_d_winner(time, op, D)
                t256 = get(time,(op,D,256), NaN)
                spd  = (isnan(t256)||isnan(w.time)||w.time==0) ? "-" :
                       string(round(t256/w.time, digits=3))
                ob   = get(occ,(op,D,w.nthreads), NaN)
                o256 = get(occ,(op,D,256), NaN)
                sched_nt = schedule[(op,D)]
                f(x) = isnan(x) ? "-" : string(x)
                fm(x)= isnan(x) ? "(not benchmarked)" :
                       string(round(x*1e3, digits=4))
                push!(detail_rows, [string(D), string(sched_nt),
                    string(w.nthreads), fm(w.time), fm(t256), spd,
                    f(ob), f(o256)])
            end
            _mdtable(io,
                ["D", "schedule nt", "argmin nthreads", "best time (ms)",
                 "256 time (ms)", "speedup", "pred. occ. best",
                 "pred. occ. 256"],
                detail_rows)
            println(io)
        end
        println(io, "## Notes\n")
        println(io, "- The schedule (`nthreads_schedule.csv`) is the ",
            "input to subsequent roofline/profile runs: the launch ",
            "harness reads `best_nthreads(op,D)` from it instead of a ",
            "hardcoded 256.")
        println(io, "- Step 1's prediction is analytic and approximate ",
            "(the raw shared-memory formula omits ~1-2 KB of driver ",
            "overhead). Cells flagged `near_boundary` in ",
            "`occupancy_prediction.csv` are where that uncertainty could ",
            "shift the predicted threshold by one D -- the benchmark, ",
            "not the prediction, is authoritative there.")
    end
    println("wrote $md")

    # ---- console summary --------------------------------------------
    println("\nchosen rules:")
    for op in OPS
        r = rules[op]
        println("  $op : ", _format_rule(r, allDs))
    end
end

isinteractive() || main()
