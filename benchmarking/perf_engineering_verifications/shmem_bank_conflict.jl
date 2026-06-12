using CSV, DataFrames, Printf

opcode(src::AbstractString) = let s = strip(src); isempty(s) ? "" : split(s)[1] end

# Strip predicate prefix ("@P0", "@!P5", ...) so we can read the bare opcode.
function bare_opcode(src::AbstractString)
    s = replace(strip(src), r"^@!?\w+\s+" => "")
    isempty(s) ? "" : split(strip(s))[1]
end

# Load SASS-page CSV. Row 1 = banner (kernel name), row 2 = column
# header, rows 3+ = per-instruction entries.
function load_sass(path)
    df = CSV.File(path; header=2, skipto=3, types=String,
                  silencewarnings=true) |> DataFrame
    # drop blank trailing rows that CSV sometimes emits
    filter!(r -> !ismissing(r.Source) && !isempty(strip(r.Source)), df)
    return df
end

parsenum(x) = (s = strip(string(x)); (isempty(s) || s == "n/a") ? 0 : parse(Int, replace(s, "," => "")))

function check_one(D::Int, op::String)
    profile_dir = joinpath(@__DIR__, "..", "roofline", op,
                             "profile_results", "tuned")
    path = joinpath(profile_dir, "$(op)_ours_D$(D)_sass.csv")
    isfile(path) || (println("  D=$D: FILE NOT FOUND ($path)"); return false)

    df = load_sass(path)
    ops = String[bare_opcode(s) for s in df.Source]

    # Nsight Compute SASS-page columns. All of these are already weighted by
    # execution count (e.g. for a conflict-free STS with Exec=N, Shared=N).
    waves_actual = Int[parsenum(x) for x in df."L1 Wavefronts Shared"]
    waves_ideal  = Int[parsenum(x) for x in df."L1 Wavefronts Shared Ideal"]
    waves_excess = Int[parsenum(x) for x in df."L1 Wavefronts Shared Excessive"]
    nway         = Int[parsenum(x) for x in df."L1 Conflicts Shared N-Way"]

    is_shared = startswith.(ops, "LDS") .| startswith.(ops, "STS") .|
                startswith.(ops, "ATOMS")

    n_excess = sum(waves_excess[is_shared])
    n_actual = sum(waves_actual[is_shared])
    n_ideal  = sum(waves_ideal[is_shared])

    # Worst N-way conflict that actually cost wavefronts. The N-Way column
    # is a pattern descriptor (how many threads target the same bank), not a
    # cost.
    # It labels broadcasts and real conflicts identically.
    # Filter to instructions where excess > 0 so the metric
    # reflects actual serialization cost.
    costly = is_shared .& (waves_excess .> 0)
    max_nway = any(costly) ? maximum(nway[costly]) : 1

    pct = n_actual == 0 ? 0.0 : 100 * n_excess / n_actual

    ok = (n_excess == 0)
    status = ok ? "OK" : "CONFLICT"
    @printf("  D=%3d  %-8s  waves=%12d  ideal=%12d  excess=%12d  conflict_pct=%6.2f%%  worst_n_way=%2d\n",
            D, status, n_actual, n_ideal, n_excess, pct, max_nway)
    return ok
end

function check_bank_conflicts(Ds, op::String)
    println("=== shared-memory bank-conflict check on $op SASS ===")
    println("    Operation: $op")
    println()
    println("    waves        = sum of L1 Wavefronts Shared (all shared accesses emitted)")
    println("    ideal        = sum of L1 Wavefronts Shared Ideal (conflict-free baseline)")
    println("    excess       = sum of L1 Wavefronts Shared Excessive (cost of conflicts)")
    println("    conflict_pct = excess / waves")
    println("    worst_n_way  = max L1 Conflicts Shared N-Way over shared instrs")
    println("                   that actually cost wavefronts (excess > 0);")
    println("                   1 = no conflict had any cost (broadcasts excluded)")
    println("    OK iff excess == 0")
    println()
    all_ok = true
    for D in Ds
        all_ok &= check_one(D, op)
    end
    println()
    if all_ok
        println("RESULT: no shared-memory bank conflicts detected in any kernel.")
    else
        println("RESULT: bank conflicts found -- inspect above output.")
    end
    return all_ok
end

Ds = collect(2:32)
for op in ["kalman", "sqrt_kalman", "gauss_likelihood"]
    check_bank_conflicts(Ds, op)
end
