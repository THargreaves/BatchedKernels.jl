#!/usr/bin/env julia
# ======================================================================
# predict_occupancy.jl  --  STEP 1 of the nthreads tuning pipeline.
#
# Predicts SM occupancy for each (op, D, nthreads), purely analytically:
#
#   * shared memory / block : the kernel's exact allocation formula
#     (see _shmem_bytes_*). The raw formula sits ~1-2 KB below the
#     profiler's launch__shared_mem_per_block (driver-reserved shared
#     memory + allocation-granularity rounding). We do NOT correct for
#     this: the prediction only needs to be approximate. Its job is to
#     locate the block-size threshold to within ~1 D; STEP 2 benchmarks
#     a range around it and makes the exact call. Cells where shmem is
#     within BOUNDARY_MARGIN of an occupancy floor-step are flagged
#     `near_boundary` -- there the +-1 D uncertainty lives, and STEP 2's
#     experiment is authoritative.
#
#   * registers / thread : NOT analytic (ptxas decides). Read from the
#     existing roofline profiles (all taken at 256 threads). Registers/
#     thread is empirically nthreads-independent -- a function of (op,D).
#
#   * hardware limits : QUERIED from the device via CUDA.jl and recorded
#     to device_info.csv, so no constant is hardcoded.
#
# occupancy = min(shared, register, warp, block) resident blocks,
#             x warps/block -> resident warps -> % of max.
#
# Outputs: results/occupancy_prediction.csv, results/device_info.csv
# ======================================================================

using CUDA
using Printf
using Dates

const TUNE_DIR    = @__DIR__
const ROOFLINE    = abspath(joinpath(TUNE_DIR, "..", "roofline"))
const RESULTS_DIR = joinpath(TUNE_DIR, "results")

# Step 1 reads registers/thread from the BASELINE roofline profiles
# (the naive 256-thread run). This is deliberate: registers/thread is
# nthreads-invariant, so the baseline profiles are a valid -- and the
# canonical -- source. Profiles live under <op>/profile_results/baseline/
# since the roofline pipeline gained the baseline|tuned split.
const PROFILE_SUBDIR = "baseline"

const OPS      = ["kalman", "sqrt_kalman", "gauss_likelihood", "qr_r"]
const DS       = collect(2:16)
const NTHREADS = [64, 96, 128, 160, 192, 224, 256]

# a cell is "near a boundary" if shmem/block is within this many bytes
# of an occupancy floor-step carveout/N -- there a ~1-2 KB formula error
# could flip the predicted block count by one.
const BOUNDARY_MARGIN = 2 * 1024

# ----------------------------------------------------------------------
# Shared memory per block, BYTES, as a function of (D, nthreads).
# Transcribed verbatim from the kernels' shared-array allocations.
# Integer divisions are floor (Julia div / ÷), matching the kernels.
# RAW formula -- intentionally uncalibrated (see header).
# ----------------------------------------------------------------------
function _shmem_bytes_gauss_likelihood(D::Int, nthreads::Int)
    n_mats_per_warp = 32 ÷ D
    n_warps         = nthreads ÷ 32
    dual_padding    = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems     = warp_shmem_size * n_warps
    shmem_vec_elems = D * n_warps * n_mats_per_warp
    return sizeof(Float32) * (2 * shmem_elems + 2 * shmem_vec_elems)
end

# kalman and sqrt_kalman share the same shared-memory layout.
function _shmem_bytes_kalman(D::Int, nthreads::Int)
    n_mats_per_warp  = 32 ÷ D
    n_warps          = nthreads ÷ 32
    dual_padding     = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    pad_interval     = div(32, D & -D) * D
    warp_shmem_size  = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems      = warp_shmem_size * n_warps
    shmem_size_fixed = D * D + (D * D - 1) ÷ pad_interval
    return sizeof(Float32) * (3 * shmem_elems + 4 * shmem_size_fixed)
end

function _shmem_bytes_qr(D::Int, nthreads::Int)
    n_mats_per_warp  = 32 ÷ D
    n_warps          = nthreads ÷ 32
    dual_padding     = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32), 32)
    warp_shmem_size  = n_mats_per_warp * D * D + dual_padding * (D - 1)
    shmem_elems      = warp_shmem_size * n_warps
    return sizeof(Float32) * 2 * shmem_elems
end

function shmem_bytes(op, D, nthreads)
    if op == "gauss_likelihood"
        return _shmem_bytes_gauss_likelihood(D, nthreads)
    elseif op in ("kalman", "sqrt_kalman")
        return _shmem_bytes_kalman(D, nthreads)
    elseif op in ("qr_r", "qr_q")
        return _shmem_bytes_qr(D, nthreads)
    end

    error("no shared-memory formula for op '$op'")
end

# ----------------------------------------------------------------------
# registers/thread from the existing roofline profile of `ours` at D.
# The roofline CSVs are `ncu --csv --page raw`: every field is wrapped in
# double quotes, and quoted fields (e.g. Kernel Name) contain commas, so
# a naive comma-split is wrong -- we parse quote-aware. Row 1 is the
# units row; the kernel row is the first with a non-empty Kernel Name.
# ----------------------------------------------------------------------

# split one CSV line into fields, honouring "double quotes" (which may
# themselves contain commas), and strip the surrounding quotes.
function _csv_fields(line::AbstractString)
    fields = String[]
    buf    = IOBuffer()
    inq    = false
    i      = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            inq = !inq
        elseif c == ',' && !inq
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function registers_per_thread(op::AbstractString, D::Int)
    path = joinpath(ROOFLINE, op, "profile_results", PROFILE_SUBDIR,
                    "$(op)_ours_D$(D).csv")
    isfile(path) || error("roofline profile not found: $path\n" *
        "  STEP 1 needs registers/thread from the overnight roofline run.")
    lines  = readlines(path)
    length(lines) < 2 && error("profile CSV has no kernel row: $path")
    header = _csv_fields(lines[1])
    ci = findfirst(==("launch__registers_per_thread"), header)
    ci === nothing &&
        error("launch__registers_per_thread column missing in $path")
    kn = findfirst(==("Kernel Name"), header)
    for li in 2:length(lines)
        f = _csv_fields(lines[li])
        length(f) < length(header) && continue
        kn !== nothing && isempty(strip(f[kn])) && continue   # units row
        r = tryparse(Int, strip(f[ci]))
        r === nothing || return r
    end
    error("no numeric registers/thread row in $path")
end

# ----------------------------------------------------------------------
# Device limits, queried from the GPU. Recorded for provenance.
#
# CUDA.jl's device-attribute enum lives in the low-level CUDA.CUDA
# submodule as CU_DEVICE_ATTRIBUTE_* constants; CUDA.attribute(dev, ·)
# reads one. We query via those, with a tiny helper so a renamed/missing
# constant fails loudly with the offending name rather than cryptically.
# ----------------------------------------------------------------------
function _attr(dev, name::Symbol)
    isdefined(CUDA.CUDA, name) ||
        error("CUDA driver attribute $name not found in this CUDA.jl " *
              "version -- check the CU_DEVICE_ATTRIBUTE_* spelling.")
    # CUDA.attribute can return Int32; coerce so all arithmetic downstream
    # (fld, comparisons) stays uniformly Int and never hits a type clash.
    return Int(CUDA.attribute(dev, getfield(CUDA.CUDA, name)))
end

function device_limits()
    dev = CUDA.device()
    max_threads_per_sm =
        _attr(dev, :CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    return (
        gpu_name              = CUDA.name(dev),
        compute_capability    = string(CUDA.capability(dev)),
        sm_count              = _attr(dev, :CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        max_threads_per_sm    = max_threads_per_sm,
        max_warps_per_sm      = max_threads_per_sm ÷ 32,
        max_blocks_per_sm     = _attr(dev, :CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR),
        regs_per_sm           = _attr(dev, :CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR),
        shmem_per_sm_optin    = _attr(dev, :CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        reg_alloc_granularity = 256,   # Ada: 256 registers per warp-allocation unit
    )
end

# nearest occupancy floor-step below/above shmem; true if within margin
function _near_boundary(shmem::Int, carveout::Int, margin::Int)
    shmem <= 0 && return false
    n = fld(carveout, shmem)            # current block count
    n <= 0 && return true
    step_lo = n   > 0 ? fld(carveout, n)     : carveout   # shmem at this n
    step_hi = n+1 > 0 ? fld(carveout, n + 1) : 0          # shmem to drop to n+1
    return (shmem - step_hi) < margin || (step_lo - shmem) < margin
end

# resident blocks per resource, then the binding minimum.
function predict(op, D, nthreads, lim)
    nwarps_per_block = nthreads ÷ 32
    shb  = shmem_bytes(op, D, nthreads)
    regs = registers_per_thread(op, D)

    occ_shmem = shb > 0 ? fld(lim.shmem_per_sm_optin, shb) : lim.max_blocks_per_sm
    regs_per_warp_alloc = cld(regs * 32, lim.reg_alloc_granularity) *
                          lim.reg_alloc_granularity
    regs_per_block = regs_per_warp_alloc * nwarps_per_block
    occ_regs  = regs_per_block > 0 ? fld(lim.regs_per_sm, regs_per_block) :
                                     lim.max_blocks_per_sm
    occ_warps = fld(lim.max_warps_per_sm, nwarps_per_block)
    occ_block = lim.max_blocks_per_sm

    resident_blocks = min(occ_shmem, occ_regs, occ_warps, occ_block)
    resident_warps  = resident_blocks * nwarps_per_block
    occ_pct         = 100 * resident_warps / lim.max_warps_per_sm

    limiter = occ_shmem  == resident_blocks ? "shared_mem" :
              occ_regs   == resident_blocks ? "registers"  :
              occ_warps  == resident_blocks ? "warps"      : "block_cap"

    near_b = _near_boundary(shb, lim.shmem_per_sm_optin, BOUNDARY_MARGIN)

    return (; shmem_per_block_B = shb, regs_per_thread = regs,
              occ_limit_shmem = occ_shmem, occ_limit_regs = occ_regs,
              occ_limit_warps = occ_warps, occ_limit_block = occ_block,
              resident_blocks, resident_warps, occ_pct, limiter,
              near_boundary = near_b)
end

function main()
    mkpath(RESULTS_DIR)
    lim = device_limits()

    open(joinpath(RESULTS_DIR, "device_info.csv"), "w") do io
        println(io, "field,value")
        for (k, v) in pairs(lim)
            println(io, "$k,$v")
        end
        println(io, "generated,$(now())")
    end
    println("device: $(lim.gpu_name)  CC $(lim.compute_capability)")
    println("  $(lim.sm_count) SMs, $(lim.max_warps_per_sm) warps/SM, " *
            "$(lim.shmem_per_sm_optin) B shared/SM (opt-in)")

    rows = NamedTuple[]
    for op in OPS, D in DS, nt in NTHREADS
        (32 ÷ D) < 1 && continue
        p = predict(op, D, nt, lim)
        push!(rows, (; op, D, nthreads = nt, p...))
    end

    out = joinpath(RESULTS_DIR, "occupancy_prediction.csv")
    open(out, "w") do io
        println(io, "op,D,nthreads,shmem_per_block_B,regs_per_thread," *
                    "occ_limit_shmem,occ_limit_regs,occ_limit_warps," *
                    "occ_limit_block,resident_blocks,resident_warps," *
                    "predicted_occupancy_pct,limiter,near_boundary")
        for r in rows
            println(io, join((r.op, r.D, r.nthreads, r.shmem_per_block_B,
                r.regs_per_thread, r.occ_limit_shmem, r.occ_limit_regs,
                r.occ_limit_warps, r.occ_limit_block, r.resident_blocks,
                r.resident_warps, round(r.occ_pct, digits=2),
                r.limiter, r.near_boundary), ","))
        end
    end
    println("wrote $out  ($(length(rows)) rows)")

    # predicted per-op threshold: smallest D at which the best nthreads
    # gives strictly more resident warps than 256.
    println("\npredicted block-size thresholds (analytic, +-1 D):")
    for op in OPS
        thr = nothing
        for D in DS
            (32 ÷ D) < 1 && continue
            base = predict(op, D, 256, lim).resident_warps
            best = maximum(predict(op, D, nt, lim).resident_warps
                           for nt in NTHREADS)
            if best > base
                thr = D; break
            end
        end
        println("  $op : ",
                thr === nothing ? "256 optimal for all D" :
                "non-256 first wins at D=$thr")
    end
    println("\n(STEP 2 benchmarks a range around each threshold and decides.)")
end

isinteractive() || main()