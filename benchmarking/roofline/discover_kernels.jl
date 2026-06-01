"""
discover_kernels.jl

NOTE: Majority of its functionalities are now legacy, not used in the pipeline,
though may still be interesting for the report for analysis on other implementations'
kernel launch details.
"""

const ROOFLINE_DIR = @__DIR__

# Per-baseline driver registry. Each entry describes how to invoke the
# counting driver for (operation, implementation). The driver MUST run
# the operation `L` times in a loop and take (D, L) as CLI args.
#   runner : "julia" or "python"
#   exe    : executable. "julia" is taken from PATH as-is; anything that
#            looks like a path is resolved relative to roofline/.
#   script : counting-driver script, relative to roofline/.
const COUNT_DRIVERS = Dict{Tuple{String,String},NamedTuple}(
    ("qr_r","jax")    => (runner="python", exe="../myenv/bin/python",
                          script="qr_r/kernel_count_jax.py"),
    ("qr_r","cublas") => (runner="julia",  exe="julia",
                          script="qr_r/kernel_count_cublas.jl"),
    ("qr_r","magma")  => (runner="julia",  exe="julia",
                          script="qr_r/kernel_count_magma.jl"),
    ("kalman","jax")   => (runner="python", exe="../myenv/bin/python",
                           script="kalman/kernel_count_jax.py"),
    ("kalman","magma") => (runner="julia",  exe="julia",
                           script="kalman/kernel_count_magma.jl"),
    ("sqrt_kalman","jax")   => (runner="python", exe="../myenv/bin/python",
                           script="sqrt_kalman/kernel_count_jax.py"),
    ("sqrt_kalman","magma") => (runner="julia",  exe="julia",
                           script="sqrt_kalman/kernel_count_magma.jl"),
    ("gauss_likelihood","jax")   => (runner="python", exe="../myenv/bin/python",
                           script="gauss_likelihood/kernel_count_jax.py"),
)

const JULIA_PROJECT_REL = "../../."

# instances/L at or above this -> treated as a per-call kernel
# below -> one-time JIT/autotuning noise. round(instances/L) = kernels-per-call.
const PERCALL_RATIO_THRESHOLD = 0.75

const MEASURE_L = 5

# discovery's own loop count
# Only used to count kernels-per-call via nsys
# (which traces the whole program cheaply)
const DISCOVERY_L = 100

# resolve a path given relative to roofline/ into an absolute path
_rel(p::AbstractString) = abspath(joinpath(ROOFLINE_DIR, p))

# Quoted-CSV row splitter. nsys quotes fields and some kernel names
# contain commas (template args), so a plain split on ',' is wrong
# track quote state and only split on unquoted commas.
function _csv_cells(line::AbstractString)
    s = strip(line)
    isempty(s) && return String[]
    cells = String[]
    buf = IOBuffer(); inq = false
    for c in s
        if c == '"'
            inq = !inq
        elseif c == ',' && !inq
            push!(cells, String(take!(buf)))
        else
            print(buf, c)
        end
    end
    push!(cells, String(take!(buf)))
    return cells
end

_num(s) = (v = tryparse(Float64, strip(s, ['"',' ',','])); v === nothing ? missing : v)

# Run nsys on the baseline driver, export the kernel summary as CSV, and
# return the CSV path. The (transient) .nsys-rep / .sqlite are stored in
# tmp_dir, only the kernel CSV matters downstream.
function _run_nsys(op, impl, D, L; tmp_dir)
    haskey(COUNT_DRIVERS, (op,impl)) ||
        error("no counting driver registered for ($op, $impl), add to COUNT_DRIVERS")
    drv = COUNT_DRIVERS[(op,impl)]
    isdir(tmp_dir) || mkpath(tmp_dir)
    # D-tagged trace basename: different D launch different kernel counts,
    # so their nsys traces must not collide in tmp/.
    rep = joinpath(tmp_dir, "$(op)_$(impl)_D$(D)_trace")

    # Resolve exe + script. nsys launches the target process itself and
    # does not reliably resolve relative paths, so anything path-like is
    # made absolute (anchored to roofline/). Bare "julia" stays on PATH.
    exe = occursin('/', drv.exe) ? _rel(drv.exe) : drv.exe
    script = _rel(drv.script)
    isfile(script) || error("counting driver script not found: $script")

    drv_cmd = if drv.runner == "julia"
        # julia project anchored to roofline/ too, so cwd doesn't matter
        proj = _rel(JULIA_PROJECT_REL)
        `$exe --project=$proj $script $D $L`
    elseif drv.runner == "python"
        `$exe $script $D $L`
    else
        error("unknown runner '$(drv.runner)'")
    end

    run(`nsys profile --force-overwrite true -o $rep $drv_cmd`)

    # `nsys stats` skips writing its CSV if the file already exists, it
    # has no --force-overwrite for the stats output. So delete any stale
    # summary CSV first, otherwise a previous run's (wrong D) summary is
    # silently reused
    csv = rep * "_cuda_gpu_kern_sum.csv"
    isfile(csv) && rm(csv)
    run(`nsys stats --force-export=true --report cuda_gpu_kern_sum
         --format csv --output $rep $(rep * ".nsys-rep")`)

    isfile(csv) || error("expected kernel-summary CSV not found: $csv")
    return csv
end

# Parse the cuda_gpu_kern_sum CSV.
# Columns: Time(%),Total Time(ns),Instances,Avg,Med,Min,Max,StdDev,Name
function _parse_kernel_csv(path)
    lines = readlines(path)
    isempty(lines) && error("empty kernel CSV: $path")
    header = _csv_cells(lines[1])
    ci_time = findfirst(==("Total Time (ns)"), header)
    ci_inst = findfirst(==("Instances"), header)
    ci_name = findfirst(==("Name"), header)
    (ci_time === nothing || ci_inst === nothing || ci_name === nothing) &&
        error("unexpected kernel CSV header in $path: $(header)")

    kernels = NamedTuple[]
    for li in 2:length(lines)
        isempty(strip(lines[li])) && continue
        c = _csv_cells(lines[li])
        length(c) < length(header) && continue
        t = _num(c[ci_time]); n = _num(c[ci_inst])
        (t isa Number && n isa Number) || continue
        push!(kernels, (name=strip(c[ci_name]), total_time_ns=t, instances=Int(n)))
    end
    isempty(kernels) && error("no kernel rows parsed from $path")
    return kernels
end

# Run discovery for one (operation, implementation).
# This is currently used in the pipeline only for the 'multi_kernel' flag
# and its existence is somewhat legacy (it was previously used to count the
# actual per operation kernel launches, before a better way was found using a marker
# dummy kernel, which is now used)
function discover_kernels(op::AbstractString, impl::AbstractString;
                          D::Int=8, L::Int=DISCOVERY_L,
                          tmp_dir::AbstractString=_rel(joinpath(op, "tmp")),
                          profile_dir::AbstractString=_rel(joinpath(op, "profile_results")),
                          write_csv::Bool=true)

    csv = _run_nsys(op, impl, D, L; tmp_dir=tmp_dir)
    kernels = _parse_kernel_csv(csv)

    # classify each kernel: per-call if instances/L >= threshold
    classified = NamedTuple[]
    for k in kernels
        ratio = k.instances / L
        per_call = ratio >= PERCALL_RATIO_THRESHOLD
        push!(classified, (
            name = k.name,
            instances = k.instances,
            total_time_ns = k.total_time_ns,
            ratio = ratio,
            per_call = per_call,
            per_call_count = per_call ? round(Int, ratio) : 0,
        ))
    end

    # per-call GPU time -> each per-call kernel's share (noise excluded)
    percall = filter(k -> k.per_call, classified)
    isempty(percall) &&
        @warn "no per-call kernels found for ($op,$impl) -- is L=$L correct, " *
              "and does the driver loop the operation L times?"
    percall_time = sum(k.total_time_ns for k in percall; init=0.0)
    kernels_per_call = sum(k.per_call_count for k in percall; init=0)

    _print_discovery(op, impl, D, L, classified, percall_time, kernels_per_call)

    if write_csv
        outdir = profile_dir
        isdir(outdir) || mkpath(outdir)
        outpath = joinpath(outdir, "$(op)_$(impl)_D$(D)_kernels.csv")
        _write_kernels_csv(outpath, op, impl, D, L, classified, percall_time)
        println("\nwrote $outpath")

        # scalars: the file measure_baseline / analyse read. The only
        # load-bearing field is multi_kernel (does this baseline need a
        # measured-bytes run?). kernels_per_call is kept as a non-binding
        # cross-check against the marker-delimited call. Launch-skip/count
        # are gone: measurement captures all kernels and uses the marker.
        scalarpath = joinpath(outdir, "$(op)_$(impl)_D$(D)_discovery.csv")
        open(scalarpath, "w") do io
            println(io, "operation,implementation,D,discovery_L," *
                        "multi_kernel,kernels_per_call")
            println(io, join((op, impl, D, L,
                length(percall) > 1, kernels_per_call), ","))
        end
        println("wrote $scalarpath")
    end

    return (
        classified = classified,
        kernels_per_call = kernels_per_call,
        n_per_call_kernels = length(percall),
        percall_time_ns = percall_time,
        multi_kernel = length(percall) > 1,
    )
end

function _print_discovery(op, impl, D, L, classified, percall_time, kpc)
    println("\n=== kernel discovery: $op / $impl  (D=$D, L=$L) ===")
    println(rpad("kernel",58), rpad("inst",7), rpad("/L",7),
            rpad("class",10), rpad("/call",7), "% per-call time")
    println("-"^104)
    for k in sort(classified; by=x->x.total_time_ns, rev=true)
        nm = length(k.name) > 56 ? k.name[1:53]*"..." : k.name
        cls = k.per_call ? "per-call" : "noise"
        pct = k.per_call && percall_time > 0 ?
              string(round(100*k.total_time_ns/percall_time, digits=1)) : "-"
        println(rpad(nm,58), rpad(k.instances,7),
                rpad(round(k.ratio,digits=2),7), rpad(cls,10),
                rpad(k.per_call ? k.per_call_count : "-",7), pct)
    end
    println("-"^104)
    npc = count(k->k.per_call, classified)
    println("per-call kernels: $npc   kernels per call: $kpc")
    if npc > 1
        println("=> MULTI-KERNEL: needs MEASURED dram bytes (measure_baseline " *
                "captures all kernels; the marker delimits one steady-state call)")
    else
        println("=> single-kernel: analytic dram bytes valid (no measured-bytes run needed)")
    end
end

function _write_kernels_csv(path, op, impl, D, L, classified, percall_time)
    open(path, "w") do io
        println(io, "operation,implementation,D,loop_count,kernel,instances," *
                    "ratio_to_L,classification,per_call_count,total_time_ns," *
                    "pct_of_per_call_time")
        for k in sort(classified; by=x->x.total_time_ns, rev=true)
            cls = k.per_call ? "per_call" : "noise"
            pct = k.per_call && percall_time > 0 ?
                  100*k.total_time_ns/percall_time : ""
            # quote the kernel name (contains commas)
            println(io, join((op, impl, D, L, "\"$(k.name)\"", k.instances,
                round(k.ratio,digits=4), cls,
                k.per_call ? k.per_call_count : "",
                k.total_time_ns, pct), ","))
        end
    end
end
