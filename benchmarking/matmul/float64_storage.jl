# Resource/correctness mode by default. MODE=timing enables throughput measurements.
include("intermediate_storage.jl")

function run_precision_storage()
    mode = get(ENV, "MODE", "resources")
    mode in ("resources", "timing") || error("MODE must be resources or timing")
    N = parse(Int, get(ENV, "BATCH", mode == "timing" ? "8193" : "5"))
    N > 0 || error("BATCH must be positive")
    cases = [
        Tuple(parse.(Int, split(x, ':'))) for
        x in split(get(ENV, "CASES", "16:64,32:32,32:64"), ',')
    ]
    types = [
        if x == "Float32"
            Float32
        elseif x == "Float64"
            Float64
        else
            error("Unsupported PRECISIONS entry")
        end for x in split(get(ENV, "PRECISIONS", "Float32,Float64"), ',')
    ]
    result_path = get(ENV, "RESULTS", joinpath(@__DIR__, "float64_$(mode)_results.csv"))
    BK.DEBUG_ACCESSORS && mode == "timing" && error("Use production accessors for timings")
    CUDA.versioninfo()
    for name in (
        :MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        :RESERVED_SHARED_MEMORY_PER_BLOCK,
        :MAX_REGISTERS_PER_MULTIPROCESSOR,
        :MAX_THREADS_PER_MULTIPROCESSOR,
        :MULTIPROCESSOR_COUNT,
    )
        println(
            "DEVICE,$name,$(CUDA.attribute(CUDA.device(),getproperty(CUDA,Symbol(:DEVICE_ATTRIBUTE_,name))))",
        )
    end
    println("RUN,mode=$mode,batch=$N,results=$result_path")
    println(
        "No register caps. Local-memory candidates are retained for this precision experiment.",
    )
    records = []
    input_checks = []
    rng = MersenneTwister(42189)
    for (D, threads) in cases
        D in (16, 32) && threads in (32, 64) ||
            error("Selected scenarios are D16/D32 and 32/64 threads")
        # Same underlying inputs for both precisions in each scenario.
        base = ntuple(_ -> randn(rng, Float64, D, D, N) / sqrt(Float64(D)), 2)
        for T in types
            inputs = map(x -> T.(x), base)
            reference = similar(inputs[1])
            for n in 1:N
                reference[:, :, n] = retained_intermediates(
                    inputs[1][:, :, n], inputs[2][:, :, n]
                )
            end
            args = Tuple(BK.BatchedCuMatrix(CuArray(x)) for x in inputs)
            push!(input_checks, (args, inputs))
            tape = BK.trace(
                retained_intermediates, BK.InputSpec[BK.input_spec(x) for x in args]
            )
            for policy in (:register, :shared_X, :shared_XY)
                a = intermediate_assignment(tape, policy, threads)
                p = BK.plan_memory(tape, a; D_MAX=D, T)
                println("PREPARE,$T,$D,$threads,$policy")
                flush(stdout)
                start = time_ns()
                entry = BK._ensure_compiled!(
                    retained_intermediates, args; assignment=a, nthreads=threads
                )
                out = similar(args[1].data)
                ka = (out, (x.data for x in args)..., Int32(N))
                kernel = Base.invokelatest() do
                    fn = entry.fn
                    @cuda launch = false fn(ka...)
                end
                compile_s = (time_ns() - start) / 1e9
                blocks = cld(N, (threads ÷ 32) * (32 ÷ D))
                CUDA.@sync kernel(ka...; threads, blocks)
                got = Array(out)
                rtol, atol = T === Float64 ? (1e-11, 1e-12) : (3e-4, 3e-5)
                # Check every batch item, not only the concatenated norm.
                @assert all(
                    isapprox(view(got, :, :, n), view(reference, :, :, n); rtol, atol) for
                    n in 1:N
                )
                relative_error =
                    maximum(abs.(got .- reference)) / max(maximum(abs, reference), eps(T))
                mem = CUDA.memory(kernel)
                regs = CUDA.registers(kernel)
                local_bytes = getproperty(mem, :local)
                active_blocks = CUDA.active_blocks(kernel.fun, threads)
                occupancy = CUDA.occupancy(kernel.fun, threads)
                status = local_bytes == 0 ? "zero_local" : "local_memory"
                println(
                    "RESOURCE,$T,$D,$threads,$policy,$regs,$local_bytes,$(mem.shared),$active_blocks,$occupancy,$relative_error",
                )
                flush(stdout)
                push!(
                    records,
                    (;
                        T,
                        D,
                        N,
                        policy,
                        kernel,
                        ka,
                        nthreads=threads,
                        blocks,
                        regs,
                        local_bytes,
                        shared=mem.shared,
                        active_blocks,
                        occupancy,
                        error=relative_error,
                        compile_s,
                        peak=p.peak_register_elements,
                        status,
                        times=Float64[],
                    ),
                )
            end
        end
    end
    if mode == "timing"
        for r in records
            kernel_time(r)
        end
        for _ in 1:9, index in randperm(rng, length(records))
            r = records[index]
            push!(r.times, kernel_time(r))
        end
    end
    @assert all(
        all(Array(x.data) == original for (x, original) in zip(args, inputs)) for
        (args, inputs) in input_checks
    )
    mkpath(dirname(abspath(result_path)))
    open(result_path, "w") do io
        println(
            io,
            "precision,D,batch,threads,policy,registers,local_bytes,shared_bytes,active_blocks,theoretical_occupancy,planner_live_elements,planner_live_32bit_words,relative_error,compile_seconds,status,median_us,p25_us,p75_us",
        )
        for r in records
            times = if isempty(r.times)
                ("", "", "")
            else
                (median(r.times), quantile(r.times, 0.25), quantile(r.times, 0.75))
            end
            fields = (
                r.T,
                r.D,
                r.N,
                r.nthreads,
                r.policy,
                r.regs,
                r.local_bytes,
                r.shared,
                r.active_blocks,
                r.occupancy,
                r.peak,
                r.peak * (sizeof(r.T) ÷ 4),
                r.error,
                r.compile_s,
                r.status,
                times...,
            )
            println(io, join(fields, ','))
            println("RESULT,", join(fields, ','))
        end
    end
    return println("INPUTS_UNCHANGED")
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_precision_storage()
