using CSV, DataFrames

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

# Parse a SASS address column ("00007d59 b3258100" or "0x728da3258100") into UInt64.
function parse_addr(s::AbstractString)
    cleaned = replace(strip(s), r"\s+" => "")
    cleaned = (startswith(cleaned, "0x") || startswith(cleaned, "0X")) ? cleaned[3:end] : cleaned
    parse(UInt64, cleaned, base=16)
end

# Strip predicate prefix ("@P0", "@!P5", "@PT", ...) and get bare opcode.
function bare_opcode(src::AbstractString)
    s = replace(strip(src), r"^@!?\w+\s+" => "")
    isempty(s) ? "" : split(strip(s))[1]
end

# Is this instruction a loop back-edge?
# BRA: classify by the literal target.
#   - Target is the last hex literal on the line.
#   - Explicit "-0x..." prefix => relative backward.
#   - Otherwise treated as absolute => compare to current address.
#
# BRX: the literal is the jump-table base, not a branch target, so the
# literal direction is meaningless. Classify by execution rate instead:
#   - A one-shot switch dispatch fires at most once per warp.
#   - A loop back-edge fires more than once per warp.
# Threshold: exec > entry_exec (entry_exec = num warps)
function is_backward_branch(src::AbstractString, current_addr::UInt64,
                             exec_count::Int, entry_exec::Int)
    op = bare_opcode(src)
    if startswith(op, "BRX")
        return exec_count > entry_exec
    end
    startswith(op, "BRA") || return false
    m = match(r"([-+]?)0x([0-9a-fA-F]+)\s*$", src)
    m === nothing && return false
    sign_str = m.captures[1]
    val = parse(UInt64, m.captures[2], base=16)
    sign_str == "-" && return true
    return val < current_addr
end

function check_one(D::Int, op::String)
    profile_dir = joinpath(@__DIR__, "..", "roofline", op,
                             "profile_results", "tuned")
    path = joinpath(profile_dir, "$(op)_ours_D$(D)_sass.csv")
    isfile(path) || (println("  D=$D: FILE NOT FOUND ($path)"); return false)

    df = load_sass(path)
    srcs  = df.Source
    addrs = UInt64[parse_addr(a) for a in df.Address]
    exec  = Int[parsenum(x) for x in df."Instructions Executed"]

    # The kernel entry instruction (df row 1) executes exactly once per warp,
    # so its exec count is the warp baseline for "executions per warp".
    entry_exec = isempty(exec) ? 0 : exec[1]

    n_static = 0
    n_executed = 0
    executed = Tuple{Int, String, Int}[]  # (row, instruction, exec count)

    for i in eachindex(srcs)
        if is_backward_branch(srcs[i], addrs[i], exec[i], entry_exec)
            n_static += 1
            if exec[i] > 0
                n_executed += 1
                push!(executed, (i, strip(srcs[i]), exec[i]))
            end
        end
    end

    ok = (n_executed == 0)
    status = ok ? "OK" : "LOOP"
    println("  D=$(lpad(D,3))  $(rpad(status,5))  " *
            "back_static=$(lpad(n_static,3))  back_executed=$(lpad(n_executed,3))")

    if !ok
        for (i, src, ex) in executed
            println("        line $i: $src   (executed $ex times)")
        end
    end
    return ok
end

function check_runtime_loops(Ds, op::String)
    println("=== runtime-loop check on $op SASS ===")
    println("    Operation: $op")
    println()
    println("    back_static   = instructions classified as loop back-edges")
    println("                    (backward BRA by literal target;")
    println("                     BRX with exec > entry_exec, i.e. >1x/warp)")
    println("    back_executed = subset that actually fire at runtime")
    println("    OK iff back_executed == 0 -- static dead branches are fine,")
    println("    and BRX used for switch dispatch (<=1x/warp) is not a loop")
    println()
    all_ok = true
    for D in Ds
        all_ok &= check_one(D, op)
    end
    println()
    if all_ok
        println("RESULT: no executed runtime loops in any kernel.")
        println("        (any nonzero static counts are backward branches in")
        println("         compiler-emitted helpers that never execute.)")
    else
        println("RESULT: real (executed) runtime loops found -- inspect above output.")
    end
    return all_ok
end

Ds = collect(2:32)
for op in ["kalman",]# "sqrt_kalman", "gauss_likelihood"]
    check_runtime_loops(Ds, op)
end