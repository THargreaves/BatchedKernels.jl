# Resource/correctness mode by default. MODE=timing enables throughput measurements.
include("hybrid_selective.jl")

# Batched inputs precede block-common inputs in the generated ABI.
covariance_four(P, H, A, Q, R) = covariance_step(P, A, Q, H, R)

function run_kalman_precision()
    mode = get(ENV, "MODE", "resources")
    mode in ("resources", "timing") || error("MODE must be resources or timing")
    N = parse(Int, get(ENV, "BATCH", mode == "timing" ? "8193" : "5"))
    N > 0 || error("BATCH must be positive")
    cases = [
        Tuple(parse.(Int, split(x, ':'))) for x in split(get(ENV, "CASES", "32:32"), ',')
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
            error("Selected Kalman scenarios use D16/D32 and 32/64 threads")
        # Same underlying inputs for both precisions in each scenario.
        baseP = Array{Float64}(undef, D, D, N)
        for n in 1:N
            x = randn(rng, D, D) / sqrt(D)
            baseP[:, :, n] = x * x' + I
        end
        common = (
            Matrix{Float64}(I, D, D) + 0.02randn(rng, D, D),
            0.2Matrix{Float64}(I, D, D),
            0.1randn(rng, D, D),
            Matrix{Float64}(I, D, D),
        )
        for T in types
            inputs = (
                T.(baseP),
                repeat(T.(common[3]), 1, 1, N),
                T.(common[1]),
                T.(common[2]),
                T.(common[4]),
            )
            reference = similar(inputs[1])
            for n in 1:N
                reference[:, :, n] = covariance_reference(
                    inputs[1][:, :, n], inputs[3], inputs[4], inputs[2][:, :, n], inputs[5]
                )
            end
            args = (
                BK.BatchedCuMatrix(CuArray(inputs[1])),
                BK.BatchedCuMatrix(CuArray(inputs[2])),
                (BK.SharedCuMatrix(CuArray(x), N) for x in inputs[3:end])...,
            )
            push!(input_checks, (args, inputs))
            tape = BK.trace(covariance_four, BK.InputSpec[BK.input_spec(x) for x in args])
            @assert BK.plan_memory(tape; order=BK.schedule(tape)).num_matrix_slots == 4
            policies =
                Symbol.(
                    split(
                        get(
                            ENV,
                            "POLICIES",
                            "register,register_row_stage,shared_predicted,shared_factor,shared_H_predicted",
                        ),
                        ',',
                    )
                )
            all(
                p -> p in (
                    :register,
                    :register_row_stage,
                    :shared_predicted,
                    :shared_factor,
                    :shared_H_predicted,
                    :shared_H_predicted_row_stage,
                    :shared_predicted_correction,
                    :shared_H_predicted_output,
                ),
                policies,
            ) || error("Unsupported POLICIES entry")
            for policy in policies
                a = selective_assignment(
                    tape,
                    if policy in (
                        :shared_H_predicted,
                        :shared_H_predicted_row_stage,
                        :shared_predicted_correction,
                        :shared_H_predicted_output,
                    )
                        :shared_predicted
                    elseif policy === :register_row_stage
                        :register
                    else
                        policy
                    end,
                    threads;
                    staging=if policy in
                               (:register_row_stage, :shared_H_predicted_row_stage)
                        :row
                    else
                        :col
                    end,
                )
                if policy in (
                    :shared_H_predicted,
                    :shared_H_predicted_row_stage,
                    :shared_H_predicted_output,
                )
                    for (id, node) in enumerate(tape.nodes)
                        node isa BK.CallNode && BK._isplacement(node.fn) || continue
                        input = tape.nodes[only(tape.nodes[only(node.args).id].args).id]
                        if input.index == 2
                            a.residences[id] = :single
                            a.orientations[only(node.args).id] = :col
                        end
                    end
                end
                # Correction is consumed through the lazy I-minus wrapper; sharing
                # its owner must therefore rebuild inherited wrapper metadata.
                products = [
                    id for (id, node) in enumerate(tape.nodes) if
                    node isa BK.CallNode && node.fn === (*)
                ]
                if policy === :shared_predicted_correction
                    a.residences[products[end - 1]] = :single
                elseif policy === :shared_H_predicted_output
                    a.residences[products[end]] = :single
                end
                # Rebuild derived wrapper metadata after changing an owner residence.
                raw = [i for (i, node) in enumerate(tape.nodes) if !(node isa BK.NewNode)]
                a = Assignment(
                    tape;
                    residences=Dict(
                        i => a.residences[i] for i in raw if haskey(a.residences, i)
                    ),
                    orientations=Dict(
                        i => a.orientations[i] for i in raw if haskey(a.orientations, i)
                    ),
                    variants=a.variants,
                    nthreads=threads,
                )
                p = BK.plan_memory(tape, a; D_MAX=D, T)
                static_limit = CUDA.attribute(
                    CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
                )
                if p.shared_bytes > static_limit
                    println(
                        "SKIP,$T,$D,$threads,$policy,static_shared_budget,planned=$(p.shared_bytes),limit=$static_limit",
                    )
                    flush(stdout)
                    continue
                end
                println(
                    "PLAN,$T,$D,$threads,$policy,single=$(p.num_single_slots),dual=$(p.num_dual_slots),shared_bound=$(p.shared_bytes),live_elements=$(p.peak_register_elements)",
                )
                println("PREPARE,$T,$D,$threads,$policy")
                flush(stdout)
                start = time_ns()
                entry = BK._ensure_compiled!(
                    covariance_four, args; assignment=a, nthreads=threads
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
    isempty(records) && error("No feasible candidates; inspect SKIP records")
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

abspath(PROGRAM_FILE) == (@__FILE__) && run_kalman_precision()
