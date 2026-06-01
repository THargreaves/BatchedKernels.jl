# ======================================================================
# benchmark_nthreads.jl  --  STEP 2 of the nthreads tuning pipeline.
#
# For each (op, D) in the tuning window, benchmarks ALL candidate
# nthreads values and records the median kernel time. No pruning: the
# benchmarks are cheap (BenchmarkTools median, no ncu) and "every
# candidate measured" is a cleaner claim than prediction-based pruning.
#
# STEP 1 only sets the D WINDOW: we benchmark from
# (predicted threshold - WINDOW_MARGIN) up to 16. Below the window,
# 256 is optimal and is recorded without benchmarking.
#
# Each operation's launch logic lives in ops/<op>.jl, loaded here into
# its own module (the three adapters all define setup_inputs /
# make_kernel / run_kernel, so they must be namespace-isolated).
#
# Output: results/nthreads_benchmarks.csv
# ======================================================================

using CUDA
using BenchmarkTools
using Printf
using JLD2

const TUNE_DIR    = @__DIR__
const RESULTS_DIR = joinpath(TUNE_DIR, "results")
const CACHE_DIR   = joinpath(TUNE_DIR, "cache")
const PRED_CSV    = joinpath(RESULTS_DIR, "occupancy_prediction.csv")
const OPS_DIR     = joinpath(TUNE_DIR, "ops")

const OPS      = ["kalman", "sqrt_kalman", "gauss_likelihood", "qr_r"]
const DS       = collect(2:32)   # full range: D bounded by warp size at 32
const NTHREADS = [64, 96, 128, 160, 192, 224, 256]
const WINDOW_MARGIN = 2          # benchmark from (threshold - margin) up
const SAMPLES       = 200        # @benchmark samples per (op,D,nthreads).
                                 # Median's standard error ~ 1/sqrt(N);
                                 # 200 is ~halved noise vs 50 at ~4x cost.
                                 # Cached per-(op,D) so paid once.

# ----------------------------------------------------------------------
# Shared-memory feasibility. At large D a (D, nthreads) cell can need
# more shared memory than the SM carveout -- kalman/sqrt_kalman carry 4
# block-resident system matrices on top of the per-warp working set, so
# e.g. D=29 does not fit at nthreads=256 (but does at <=128). Such cells
# must be SKIPPED before launch, or the kernel launch fails.
#
# The check is per-(op,D,nthreads), not a flat D limit: D=29..32 remain
# reachable at smaller block sizes. shmem_bytes mirrors the formulas in
# predict_occupancy.jl (canonical copy there) and each kernel's own
# launch code.
const SHMEM_CARVEOUT = 101376    # RTX 4090 opt-in max, bytes (99 KB).
                                 # device query reported 102400; 101376
                                 # is the conservative published max --
                                 # using the smaller value never lets an
                                 # infeasible cell through.

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

# true if (op, D, nthreads) fits one block into the SM carveout.
_shmem_feasible(op, D, nt) = _shmem_bytes(op, D, nt) <= SHMEM_CARVEOUT

# ----------------------------------------------------------------------
# load each operation's adapter into its own module so the three
# setup_inputs/make_kernel/run_kernel triples do not collide.
# ----------------------------------------------------------------------
module KalmanOp
    include(joinpath(@__DIR__, "ops", "kalman.jl"))
end
module SqrtKalmanOp
    include(joinpath(@__DIR__, "ops", "sqrt_kalman.jl"))
end
module GaussLikelihoodOp
    include(joinpath(@__DIR__, "ops", "gauss_likelihood.jl"))
end
module QRROp
    include(joinpath(@__DIR__, "ops", "qr_r.jl"))
end
# module QRQOp
#     include(joinpath(@__DIR__, "ops", "qr_q.jl"))
# end

const ADAPTER = Dict(
    "kalman"           => KalmanOp,
    "sqrt_kalman"      => SqrtKalmanOp,
    "gauss_likelihood" => GaussLikelihoodOp,
    "qr_r"             => QRROp,
    # "qr_q"             => QRQOp,
)

# ----------------------------------------------------------------------
# STEP 1's predicted threshold per op: smallest D where some non-256
# nthreads gives strictly more resident warps than 256.
# ----------------------------------------------------------------------
function predicted_thresholds()
    isfile(PRED_CSV) || error("$PRED_CSV not found -- run STEP 1 first.")
    lines  = readlines(PRED_CSV)
    header = split(strip(lines[1]), ",")
    ci(c)  = findfirst(==(c), header)
    iop, iD, intd, irw =
        ci("op"), ci("D"), ci("nthreads"), ci("resident_warps")
    rw = Dict{Tuple{String,Int,Int},Int}()
    for li in 2:length(lines)
        f = split(strip(lines[li]), ",")
        length(f) < length(header) && continue
        rw[(String(f[iop]), parse(Int,f[iD]), parse(Int,f[intd]))] =
            parse(Int, f[irw])
    end
    thr = Dict{String,Union{Int,Nothing}}()
    for op in OPS
        Ds = sort(unique(d for (o,d,_) in keys(rw) if o == op))
        t  = nothing
        for D in Ds
            base = get(rw, (op,D,256), 0)
            best = maximum(get(rw,(op,D,nt),0) for nt in NTHREADS)
            if best > base
                t = D; break
            end
        end
        thr[op] = t
    end
    return thr
end

# benchmark all NTHREADS for one (op, D). Returns Vector{(nthreads, time_s)}.
# Cells that exceed the shared-memory carveout are skipped (not all
# (D, nthreads) pairs fit at large D); the returned vector simply omits
# them, so an infeasible cell is absent rather than zero.
function bench_one_D(op::AbstractString, D::Integer)
    mod   = ADAPTER[op]
    state = Base.invokelatest(mod.setup_inputs, D)
    out   = Tuple{Int,Float64}[]
    for nt in NTHREADS
        (32 ÷ D) < 1 && continue
        if !_shmem_feasible(op, D, nt)
            @printf("  nt=%-3d  skipped (shared memory exceeds carveout)\n", nt)
            continue
        end
        # recompile per nthreads -- Val{nthreads} is a compile-time param.
        kernel, cfg = Base.invokelatest(mod.make_kernel, D, nt, state)
        Base.invokelatest(mod.run_kernel, kernel, cfg, state)   # warm-up
        CUDA.synchronize()
        bench = @benchmark(
            Base.invokelatest($mod.run_kernel, $kernel, $cfg, $state),
            samples = SAMPLES, evals = 1)
        tmed = median(bench).time / 1e9
        push!(out, (nt, tmed))
        @printf("  nt=%-3d  %.6f ms\n", nt, tmed * 1e3)
    end
    return out
end

# cache file for one (op, D)
_cache_path(op, D) = joinpath(CACHE_DIR, "bench_$(op)_D$(D).jld2")

# load cached (op,D) timings, or benchmark and cache them.
# `force` skips the cache entirely (and overwrites it).
function cached_bench(op::AbstractString, D::Integer, force::Bool)
    path = _cache_path(op, D)
    if !force && isfile(path)
        data = JLD2.load(path)
        # validate the cache matches the current NTHREADS set
        if get(data, "nthreads_set", nothing) == NTHREADS
            println("--- $op  D=$D  (cached)")
            return Vector{Tuple{Int,Float64}}(data["timings"])
        else
            println("--- $op  D=$D  (cache stale: NTHREADS changed, redo)")
        end
    end
    println("--- $op  D=$D ---")
    timings = bench_one_D(op, D)
    JLD2.jldsave(path; timings = timings, nthreads_set = NTHREADS,
                 op = op, D = D)
    return timings
end

function main()
    mkpath(RESULTS_DIR)
    mkpath(CACHE_DIR)
    force = get(ENV, "TUNE_FORCE", "0") == "1"
    force && println("force=true: ignoring benchmark cache")

    thr = predicted_thresholds()
    println("STEP 1 predicted thresholds: ", thr)

    out = joinpath(RESULTS_DIR, "nthreads_benchmarks.csv")
    open(out, "w") do io
        println(io, "op,D,nthreads,median_time_s,batch_size,benchmarked")

        for op in OPS
            t   = thr[op]
            dlo = t === nothing ? 17 : max(2, t - WINDOW_MARGIN)

            for D in DS
                (32 ÷ D) < 1 && continue
                N = _batch_size(op, D)

                if D < dlo
                    # below the window: 256 optimal, recorded untimed.
                    println(io, join((op, D, 256, "", N, false), ","))
                    flush(io)
                    continue
                end

                # cached or freshly benchmarked -- the JLD2 cache is the
                # source of truth; the CSV is always rebuilt from it.
                timings = cached_bench(op, D, force)
                for (nt, tmed) in timings
                    println(io, join((op, D, nt, tmed, N, true), ","))
                end
                flush(io)
            end
        end
    end
    println("\nwrote $out")
end

# batch size N per op (only needed for the untimed below-window rows).
# Matches flops.jl's batch_size_* exactly.
function _batch_size(op::AbstractString, D::Integer)
    op == "gauss_likelihood" && return Int(ceil(1e9 / (4 * 1 * D^2)))
    return Int(ceil(1e9 / (4 * 2 * D^2)))       # kalman / sqrt_kalman / QR
end

isinteractive() || main()
