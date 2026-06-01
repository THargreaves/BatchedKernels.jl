using CSV, DataFrames

opcode(src::AbstractString) = let s = strip(src); isempty(s) ? "" : split(s)[1] end

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

    df  = load_sass(path)
    ops = String[opcode(s) for s in df.Source]
    exec = Int[parsenum(x) for x in df."Instructions Executed"]

    # Static counts (presence in SASS)
    is_ldl = startswith.(ops, "LDL")
    is_stl = startswith.(ops, "STL")
    n_ldl_static = count(is_ldl)
    n_stl_static = count(is_stl)

    # Dynamic counts (actually executed, these are the real spills)
    # Kernels often have dead, unexecuted local memory code for errors
    n_ldl_executed = sum(exec[is_ldl])
    n_stl_executed = sum(exec[is_stl])

    ok = (n_ldl_executed == 0 && n_stl_executed == 0)
    status = ok ? "OK" : "SPILL"
    println("  D=$(lpad(D,3))  $(rpad(status,6))  " *
            "LDL_static=$(lpad(n_ldl_static,3))  STL_static=$(lpad(n_stl_static,3))  " *
            "LDL_exec=$(lpad(n_ldl_executed,10))  STL_exec=$(lpad(n_stl_executed,10))")
    return ok
end

function check_local_spills(Ds, op::String)
    println("=== local-memory spill check on $op SASS ===")
    println("    Operation: $op")
    println()
    println("    LDL/STL = load/store local memory (spill markers)")
    println("    any nonzero count => register spilling occurs")
    println()
    all_ok = true
    for D in Ds
        all_ok &= check_one(D, op)
    end
    println()
    if all_ok
        println("RESULT: no local-memory spilling detected in any kernel.")
    else
        println("RESULT: local-memory spills found -- inspect above output.")
    end
    return all_ok
end

Ds = collect(2:32)
for op in ["kalman", "sqrt_kalman", "gauss_likelihood"]
    check_local_spills(Ds, op)
end
