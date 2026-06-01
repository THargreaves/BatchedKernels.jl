
const UNIT_SCALE = Dict(
    ""        => 1.0,
    "byte"    => 1.0, "Kbyte" => 1e3, "Mbyte" => 1e6, "Gbyte" => 1e9,
    "byte/s"  => 1.0, "Kbyte/s" => 1e3, "Mbyte/s" => 1e6, "Gbyte/s" => 1e9,
    "ns"      => 1e-9, "us" => 1e-6, "ms" => 1e-3, "s" => 1.0,
    "%"       => 1.0,
    "inst"    => 1.0, "warp" => 1.0, "block" => 1.0, "cycle" => 1.0,
    "register/thread" => 1.0, "byte/block" => 1.0, "Kbyte/block" => 1e3,
    "Ghz" => 1e9, "Mhz" => 1e6, "Khz" => 1e3, "hz" => 1.0,
    "sector/ns" => 1e9,
    "SM" => 1.0,
    "inst/cycle" => 1.0, "sector" => 1.0, "sectors" => 1.0,
    "byte/sector" => 1.0, "thread" => 1.0, "request" => 1.0,
    "byte/cycle" => 1.0, "sector/cycle" => 1.0, "sector/s" => 1.0,
    "pass" => 1.0, "sample" => 1.0, "branches" => 1.0,
    "thread/warp" => 1.0, "register" => 1.0, "inst/warp" => 1.0,
    "cycle/warp" => 1.0, "byte/1" => 1.0, "max_rate" => 1.0,
)

_WARNED_UNITS = Set{String}()

function _scale(unit::AbstractString)
    u = strip(unit)
    haskey(UNIT_SCALE, u) && return UNIT_SCALE[u]
    if !(u in _WARNED_UNITS)
        push!(_WARNED_UNITS, u)
        @warn "unknown unit '$u' -- treating as scale 1.0; add to UNIT_SCALE if you read this metric"
    end
    return 1.0
end

function _num(s::AbstractString)
    t = replace(strip(s, ['"', ' ']), "," => "")
    isempty(t) && return missing
    v = tryparse(Float64, t)
    return v === nothing ? missing : v
end

"""
    parse_ncu_csv(path) -> Vector{Dict{String,Any}}

Parse one ncu csv file. One Dict per kernel row (non-fused
baselines launch multiple kernels -> multiple rows). Numeric values are
normalized to base SI, non-numeric fields kept as strings.
"""
function parse_ncu_csv(path::AbstractString)
    raw = readlines(path)
    start = findfirst(l -> startswith(l, "\"ID\""), raw)
    start === nothing && error("no header row found in $path")
    lines = raw[start:end]

    function cells(l)
        s = l
        startswith(s, "\"") && (s = s[2:end])
        endswith(s, "\"")   && (s = s[1:end-1])
        return String.(split(s, "\",\""))
    end

    header = cells(lines[1])
    units  = length(lines) >= 2 ? cells(lines[2]) : fill("", length(header))
    rows   = Dict{String,Any}[]
    for li in 3:length(lines)
        isempty(strip(lines[li])) && continue
        vals = cells(lines[li])
        length(vals) != length(header) && continue
        d = Dict{String,Any}()
        for (j, h) in enumerate(header)
            num = _num(vals[j])
            if num === missing
                d[h] = strip(vals[j], ['"', ' '])
            else
                d[h] = num * _scale(get(units, j, ""))
            end
        end
        push!(rows, d)
    end
    isempty(rows) && error("no data rows parsed from $path")
    return rows
end

function _sum_metric(rows, key)
    tot = 0.0; found = false
    for r in rows
        v = get(r, key, missing)
        if v isa Number
            tot += v; found = true
        end
    end
    return found ? tot : missing
end

function _first_metric(rows, key)
    for r in rows
        v = get(r, key, missing)
        v isa Number && return v
    end
    return missing
end
