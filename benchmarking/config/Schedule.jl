# ======================================================================
# benchmarking/config/Schedule.jl
#
# Single source of truth for the tuned block size (nthreads). The tuning
# pipeline (tune_nthreads/) writes config/nthreads_schedule.csv;
# every consumer -- roofline, benchmarks, any other study -- reads the
# block size through best_nthreads(op, D) here, so there is exactly one
# place the schedule is interpreted.
#
# usage:
#   include(".../config/Schedule.jl")
#   nt = Schedule.best_nthreads("kalman", 14)   # -> 128 (or whatever)
#
# Operations NOT in the schedule (matmul, cholesky, qr_r, trig_backsolve)
# fall back to DEFAULT_NTHREADS = 256: best_nthreads is total, so callers
# never special-case "is this op tuned".
# ======================================================================
module Schedule

const DEFAULT_NTHREADS = 256

# config/nthreads_schedule.csv lives next to this file.
const SCHEDULE_CSV = joinpath(@__DIR__, "nthreads_schedule.csv")

# (op, D) -> best_nthreads, loaded lazily and cached.
const _TABLE = Ref{Union{Nothing,Dict{Tuple{String,Int},Int}}}(nothing)

function _load()
    tbl = Dict{Tuple{String,Int},Int}()
    if !isfile(SCHEDULE_CSV)
        @warn "Schedule: $SCHEDULE_CSV not found; best_nthreads will " *
              "return DEFAULT_NTHREADS ($DEFAULT_NTHREADS) for every op. " *
              "Run benchmarking/tune_nthreads/run_tuning.sh to generate it."
        return tbl
    end
    lines  = readlines(SCHEDULE_CSV)
    isempty(lines) && return tbl
    header = split(strip(lines[1]), ",")
    iop = findfirst(==("op"), header)
    iD  = findfirst(==("D"),  header)
    int = findfirst(==("best_nthreads"), header)
    (iop === nothing || iD === nothing || int === nothing) &&
        error("Schedule: $SCHEDULE_CSV missing op/D/best_nthreads columns")
    for li in 2:length(lines)
        f = split(strip(lines[li]), ",")
        length(f) < length(header) && continue
        op = String(strip(f[iop]))
        D  = tryparse(Int, strip(f[iD]))
        nt = tryparse(Int, strip(f[int]))
        (D === nothing || nt === nothing) && continue
        tbl[(op, D)] = nt
    end
    return tbl
end

_table() = (_TABLE[] === nothing && (_TABLE[] = _load()); _TABLE[])

"""
    best_nthreads(op, D) -> Int

Block size for `op` at dimension `D` from the tuning schedule. Operations
not in the schedule (or if the schedule file is absent) fall back to
DEFAULT_NTHREADS = 256.
"""
best_nthreads(op::AbstractString, D::Integer) =
    get(_table(), (String(op), Int(D)), DEFAULT_NTHREADS)

"Force a re-read of the schedule CSV (e.g. after re-running the tuner)."
reload!() = (_TABLE[] = nothing; _table(); nothing)

end # module
