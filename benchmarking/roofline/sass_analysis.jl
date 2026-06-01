"""
sass_analysis.jl

Per-instruction analysis from an ncu source-page CSV export. Two analyses:

  bank_conflicts(sass_csv)   -- shared-memory bank-conflict summary
  instruction_mix(sass_csv)  -- dynamic instruction mix (opcode ranking)

WHY THIS EXISTS
  The device-aggregate counter `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`
  (raw-page CSV) was found to be UNRELIABLE: for matmul D=8 it reported
  ~1.6% conflicts, while the per-instruction source-page data shows ZERO.
  The per-instruction "L1 Wavefronts Shared Excessive" column is the
  trustworthy figure -- source-correlated, no cross-partition aggregation.

INPUT
  Produced (no re-profiling -- a second export of the same .ncu-rep) by:
    ncu --import <op>_<impl>_D<d>.ncu-rep --page source --print-source sass \\
        --csv > <op>_<impl>_D<d>_sass.csv

  Format: line 1 is a kernel-name banner, line 2 the real header, then one
  row per SASS instruction.
"""

# Minimal quoted-CSV row splitter. ncu wraps every field in double quotes
# and fields contain commas (e.g. instruction text), so a plain split on
# ',' is wrong -- split on '","' and strip the outer quotes.
function _csv_cells(line::AbstractString)
    s = strip(line)
    isempty(s) && return String[]
    startswith(s, "\"") && (s = s[2:end])
    endswith(s, "\"")   && (s = s[1:end-1])
    return String.(split(s, "\",\""))
end

# parse a numeric cell; ncu uses "-" or "" for not-applicable
function _sass_num(s::AbstractString)
    t = strip(s, ['"', ' '])
    (isempty(t) || t == "-") && return missing
    v = tryparse(Float64, replace(t, "," => ""))
    return v
end

# locate the real header row (starts with the "Address" field) and return
# (header::Vector{String}, first_data_line_index)
function _sass_header(lines)
    hidx = findfirst(l -> startswith(strip(l), "\"Address\""), lines)
    hidx === nothing && error("no header row found in SASS CSV")
    return _csv_cells(lines[hidx]), hidx
end

# ======================================================================
# Bank conflicts
# ======================================================================
"""
    bank_conflicts(sass_csv_path) -> NamedTuple

Summarise shared-memory bank conflicts from per-instruction source-page data.

Relevant columns:
  "Address Space"                   "Shared" for LDS/STS
  "L1 Conflicts Shared N-Way"
  "L1 Wavefronts Shared Excessive"
  "L1 Wavefronts Shared"
  "L1 Wavefronts Shared Ideal"

  bank_conflict_pct = 100 * (sum Excessive) / (sum Shared)
  i.e. the fraction of all shared-memory wavefront traffic that was conflict
  overhead. Summed before dividing so an instruction executed many times is
  weighted by its execution count.

Returns:
  total_excessive, total_shared, conflict_pct, max_nway, n_shared_insts,
  conflicting : Vector of (address, instruction, nway, excessive) with
                Excessive > 0  (empty if conflict-free)
"""
function bank_conflicts(sass_csv_path::AbstractString)
    isfile(sass_csv_path) || error("SASS CSV not found: $sass_csv_path")
    lines = readlines(sass_csv_path)
    header, hidx = _sass_header(lines)

    col(name) = begin
        i = findfirst(==(name), header)
        i === nothing && error("column '$name' not in $sass_csv_path")
        i
    end
    c_addr   = col("Address")
    c_instr  = col("Source")
    c_space  = col("Address Space")
    c_op     = col("Access Operation")
    c_nway   = col("L1 Conflicts Shared N-Way")
    c_exc    = col("L1 Wavefronts Shared Excessive")
    c_shared = col("L1 Wavefronts Shared")
    c_ideal  = col("L1 Wavefronts Shared Ideal")

    total_exc = 0.0
    total_shr = 0.0
    max_nway  = 0.0
    n_shared  = 0
    conflicting = Tuple{String,String,Float64,Float64}[]

    for li in (hidx + 1):length(lines)
        cells = _csv_cells(lines[li])
        length(cells) < length(header) && continue
        strip(cells[c_space], ['"', ' ']) == "Shared" || continue

        nway  = _sass_num(cells[c_nway])
        exc   = _sass_num(cells[c_exc])
        shr   = _sass_num(cells[c_shared])
        ideal = _sass_num(cells[c_ideal])
        (exc isa Number && shr isa Number) || continue

        n_shared += 1
        total_exc += exc
        total_shr += shr
        # N-Way is a pattern descriptor, not a cost.
        # broadcasts (many threads hit the same address, served free
        # in one wavefront) get the same N-Way label as real conflicts.
        # Only count N-Way from instructions that actually cost
        # wavefronts so max_nway agrees with total_exc.
        (nway isa Number && exc > 0) && (max_nway = max(max_nway, nway))

        if ideal isa Number && abs((shr - ideal) - exc) > 0.5
            @warn "SASS row inconsistency at $(cells[c_addr]): " *
                  "shared-ideal=$(shr-ideal) but excessive=$exc"
        end

        if exc > 0
            push!(conflicting, (
                strip(cells[c_addr], ['"', ' ']),
                strip(cells[c_instr], ['"', ' ']),
                nway isa Number ? nway : NaN,
                exc,
            ))
        end
    end

    n_shared == 0 && error("no shared-memory instructions found in $sass_csv_path")
    conflict_pct = total_shr > 0 ? 100.0 * total_exc / total_shr : 0.0

    if (max_nway <= 1.0) != (total_exc == 0.0)
        @warn "bank-conflict signals disagree: max_nway=$max_nway, " *
              "total_excessive=$total_exc (possible column misparse)"
    end

    return (
        total_excessive = total_exc,
        total_shared    = total_shr,
        conflict_pct    = conflict_pct,
        max_nway        = max_nway,
        n_shared_insts  = n_shared,
        conflicting     = conflicting,
    )
end

# ======================================================================
# Instruction mix
# ======================================================================
# Reduce a raw SASS instruction string to its opcode "family".
#   "@!P0  STS.128 [R23+0x10], R34"  ->  "STS"
# Steps: drop a leading predicate token (@P0 / @!P6 / @PT), take the first
# remaining token, drop any ".suffix" (.128, .E, .SYNC, .U32, ...).
# This is operation-agnostic: it just buckets by opcode.
function _opcode(instr::AbstractString)
    toks = split(strip(instr))
    isempty(toks) && return "?"
    # a predicate token starts with '@'
    idx = (startswith(toks[1], "@")) ? 2 : 1
    idx > length(toks) && return "?"
    op = toks[idx]
    dot = findfirst('.', op)
    return dot === nothing ? String(op) : String(op[1:dot-1])
end

"""
    instruction_mix(sass_csv_path; top_n=8) -> NamedTuple

Rank SASS opcodes by DYNAMIC execution count (the "Instructions Executed"
column: how many times the instruction actually ran, summed over the
grid), not by static listing count. This is the operation-agnostic
"what is this kernel spending instructions on" diagnostic.
E.g., for a shuffle-heavy kernel SHFL rises to the top

Returns:
  total_executed   : sum of "Instructions Executed" over all instructions
  ranked           : Vector of (opcode, executed, pct) sorted descending,
                      pct = 100 * executed / total_executed
  dominant         : opcode with the highest execution count
  dominant_pct     : its share of total executed instructions
  top              : the first `top_n` entries of `ranked`
"""
function instruction_mix(sass_csv_path::AbstractString; top_n::Int=8)
    isfile(sass_csv_path) || error("SASS CSV not found: $sass_csv_path")
    lines = readlines(sass_csv_path)
    header, hidx = _sass_header(lines)

    c_instr = findfirst(==("Source"), header)
    c_exec  = findfirst(==("Instructions Executed"), header)
    (c_instr === nothing || c_exec === nothing) &&
        error("Source / Instructions Executed columns not in $sass_csv_path")

    counts = Dict{String,Float64}()
    for li in (hidx + 1):length(lines)
        cells = _csv_cells(lines[li])
        length(cells) < length(header) && continue
        ex = _sass_num(cells[c_exec])
        (ex isa Number && ex > 0) || continue          # skip non-executed lines
        op = _opcode(cells[c_instr])
        counts[op] = get(counts, op, 0.0) + ex
    end
    isempty(counts) && error("no executed instructions found in $sass_csv_path")

    total = sum(values(counts))
    ranked = sort([(op, ex, 100.0 * ex / total) for (op, ex) in counts];
                  by = x -> x[2], rev = true)

    return (
        total_executed = total,
        ranked         = ranked,
        dominant       = ranked[1][1],
        dominant_pct   = ranked[1][3],
        top            = ranked[1:min(top_n, length(ranked))],
    )
end

# compact "OP:pct,OP:pct,..." string for the tidy CSV
function instruction_mix_summary(mix; n::Int=5)
    parts = ["$(op):$(round(pct, digits=1))" for (op, _, pct) in mix.top[1:min(n, length(mix.top))]]
    return join(parts, ";")
end
