# Run with julia --project=. benchmarking/kalman/hybrid_m6.jl
# Production preferences must be active; debug builds are correctness-only.
using BatchedKernels, CUDA, LinearAlgebra, Random, Statistics, Printf
const BK = BatchedKernels

function covariance_step(P, A, Q, H, R)
    predicted = A * P * A' + Q
    rhs = H * predicted
    U = cholesky(rhs * H' + R).U
    gain_t = U \ (U' \ rhs)
    return (I - gain_t' * H) * predicted
end
function covariance_reference(P, A, Q, H, R)
    predicted = A * P * A' + Q
    rhs = H * predicted
    # Roundoff can make the dense innovation differ across triangles; like the
    # GPU Cholesky body, the CPU reference uses its upper triangle.
    U = cholesky(Symmetric(rhs * H' + R, :U)).U
    gain_t = U \ (U' \ rhs)
    return (I - gain_t' * H) * predicted
end
matmul_probe(A, B) = A * B
solve_probe(A, B) = UpperTriangular(A) \ B

function forced_assignment(tape, policy; nthreads=128, orientation=:row)
    residences = Dict{Int,Symbol}()
    orientations = Dict{Int,Symbol}()
    variants = Dict{Int,Symbol}()
    for (id, node) in enumerate(tape.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            residences[id] = policy in (:input_register, :register, :balanced_mul, :balanced_factor) ? :register : :single
            orientations[id] = orientation
            orientations[only(node.args).id] = orientation
        elseif node.fn in (*, +, -, cholesky, (\))
            factor = node.fn in (cholesky, (\))
            residence = if policy === :register || (policy === :balanced_mul && !factor) || (policy === :balanced_factor && factor)
                :register
            else
                :single
            end
            residences[id], orientations[id] = residence, orientation
            prefix = node.fn === (*) ? :matmul : node.fn === (+) ? :add : node.fn === (-) ? :sub : node.fn === cholesky ? :cholesky : :solve
            variants[id] = Symbol(prefix, :_, orientation)
        end
    end
    policy === :dual && return Assignment(tape; nthreads)
    return Assignment(tape; residences, orientations, variants, nthreads)
end

function prepare(f, args, policy, reference; nthreads=128, orientation=:row, assignment_override=nothing)
    tape = BK.trace(f, BK.InputSpec[BK.input_spec(x) for x in args])
    assignment = policy === :legacy ? nothing : forced_assignment(tape, policy; nthreads, orientation)
    assignment_override === nothing || (assignment = assignment_override)
    output = similar(first(args).data, size(reference))
    println("PREPARE,$(nameof(f)),$policy,$nthreads,$orientation"); flush(stdout)
    t0 = time_ns()
    entry = BK._ensure_compiled!(f, args; assignment, nthreads)
    ka = (output, (x.data for x in args)..., Int32(size(reference, 3)))
    kernel = Base.invokelatest() do
        fn = entry.fn
        @cuda launch=false fn(ka...)
    end
    compile_s = (time_ns() - t0) / 1e9
    memory = CUDA.memory(kernel)
    local_bytes = getproperty(memory, :local)
    regs = CUDA.registers(kernel)
    occupancy = CUDA.occupancy(kernel.fun, nthreads)
    blocks = cld(size(reference, 3), (nthreads ÷ 32) * (32 ÷ entry.D_MAX))
    CUDA.@sync kernel(ka...; threads=nthreads, blocks)
    got = Array(output)
    error = maximum(abs.(got .- reference)) / max(maximum(abs, reference), eps(Float32))
    correct = isapprox(got, reference; rtol=3f-4, atol=3f-5)
    peak = assignment === nothing ? -1 : BK.plan_memory(tape, assignment; D_MAX=entry.D_MAX).peak_register_elements
    status = !correct ? "incorrect" : local_bytes > 0 ? "spilling" : "admitted"
    @printf("ADMISSION,%s,%s,%d,%d,%d,%d,%d,%d,%.4f,%.6g,%.3f,%s\n", nameof(f), policy, entry.D_MAX, size(reference,3), nthreads, regs, memory.shared, local_bytes, occupancy, error, compile_s, status)
    flush(stdout)
    status == "admitted" || return nothing
    return (; f, args, assignment, kernel, ka, blocks, nthreads, policy, D=entry.D_MAX, N=size(reference,3), regs, shared=memory.shared, local_bytes, occupancy, peak, compile_s, error, times=Float64[], endtimes=Float64[])
end

function kernel_time(r)
    CUDA.synchronize()
    return 1e6 * CUDA.@elapsed(begin
        for _ in 1:20
            r.kernel(r.ka...; threads=r.nthreads, blocks=r.blocks)
        end
    end) / 20
end
function end_time(r)
    CUDA.synchronize()
    t0 = time_ns()
    result = if r.policy === :prototype
        out = similar(r.ka[1])
        r.kernel(out, r.ka[2:end]...; threads=r.nthreads, blocks=r.blocks)
        out
    else
        fuse(r.f, r.args...; assignment=r.assignment, nthreads=r.nthreads)
    end
    CUDA.synchronize()
    return (time_ns() - t0) / 1e3
end
function measure!(records, rng)
    for r in records
        kernel_time(r)
        end_time(r)
    end
    for _ in 1:9, index in randperm(rng, length(records))
        r = records[index]
        push!(r.times, kernel_time(r))
        push!(r.endtimes, end_time(r))
    end
    for r in records
        # Logical batch I/O only, not measured DRAM or shared-memory traffic.
        logical_bytes = sum(length(x.data)*sizeof(Float32) for x in r.args if x isa BK.BatchedCuMatrix) + length(r.ka[1])*sizeof(Float32)
        @printf("RESULT,%s,%s,%d,%d,%d,%d,%d,%d,%.4f,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", nameof(r.f), r.policy, r.D, r.N, r.nthreads, r.regs, r.shared, r.local_bytes, r.occupancy, r.peak, median(r.times), quantile(r.times,.25), quantile(r.times,.75), median(r.endtimes), logical_bytes / median(r.times) / 1000)
    end
    flush(stdout)
end

# Evaluate only the existing kernel definitions; its unrelated timing helper needs
# BenchmarkTools. The prototype kernel and its helper bodies are unchanged.
module RegisterPrototype
    source = read(joinpath(@__DIR__, "../benchmarks/bench_kalman/ours_register.jl"), String)
    definitions = first(split(source, "function kalman_timing("; limit=2))
    include_string(@__MODULE__, replace(definitions, "using BenchmarkTools" => ""), "ours_register_kernel.jl")
end
function prepare_prototype(args, reference)
    # The historical loader assigns four inputs to warps 1:4 without cycling.
    nthreads = 256
    D, _, N = size(reference)
    out = similar(first(args).data)
    ka = (out, (x.data for x in args)..., Val(Int32(D)), Val(Int32(nthreads)), Int32(N))
    t0 = time_ns()
    kernel = @cuda launch=false RegisterPrototype.kernel_kalman_register!(ka...)
    compile_s = (time_ns()-t0)/1e9
    mem = CUDA.memory(kernel)
    regs, local_bytes = CUDA.registers(kernel), getproperty(mem,:local)
    blocks = cld(N,(nthreads÷32)*(32÷D))
    CUDA.@sync kernel(ka...;threads=nthreads,blocks)
    got = Array(out)
    error = maximum(abs.(got .- reference)) / maximum(abs,reference)
    correct = isapprox(got,reference;rtol=3f-4,atol=3f-5)
    status = !correct ? "incorrect" : local_bytes > 0 ? "spilling" : "admitted"
    occupancy = CUDA.occupancy(kernel.fun,nthreads)
    @printf("ADMISSION,covariance_step,prototype,%d,%d,%d,%d,%d,%d,%.4f,%.6g,%.3f,%s\n",D,N,nthreads,regs,mem.shared,local_bytes,occupancy,error,compile_s,status)
    flush(stdout)
    status == "admitted" || return nothing
    return (;f=covariance_step,args,assignment=nothing,kernel,ka,blocks,nthreads,policy=:prototype,D,N,regs,shared=mem.shared,local_bytes,occupancy,peak=-1,compile_s,error,times=Float64[],endtimes=Float64[])
end

function run_benchmarks()
    CUDA.functional() || error("CUDA is required")
    BK.DEBUG_ACCESSORS && error("Use production debug_accessors=false for timing")
    CUDA.versioninfo()
    println("ADMISSION,workload,policy,D,N,threads,registers,shared_bytes,local_bytes,occupancy,relative_max_error,compile_s,status")
    println("RESULT,workload,policy,D,N,threads,registers,shared_bytes,local_bytes,occupancy,peak_register_elements,kernel_us,q25_us,q75_us,end_to_end_us,logical_GBs")
    rng = MersenneTwister(62026)
    N = parse(Int, get(ENV, "BK_M6_N", "8192"))
    dims = parse.(Int, split(get(ENV,"BK_M6_DIMS","8,16,32"), ','))
    for D in dims
        P = Array{Float32}(undef,D,D,N)
        for n in 1:N
            X = randn(rng, Float32,D,D) / sqrt(Float32(D))
            P[:,:,n] = X * X' + I
        end
        A = Matrix{Float32}(I,D,D) + .02f0 * randn(rng,Float32,D,D)
        H = .1f0 * randn(rng,Float32,D,D)
        Q = .2f0 * Matrix{Float32}(I,D,D)
        R = Matrix{Float32}(I,D,D)
        reference = similar(P)
        for n in 1:N
            reference[:,:,n] = covariance_reference(P[:,:,n],A,Q,H,R)
        end
        args = (BK.BatchedCuMatrix(CuArray(P)), (BK.SharedCuMatrix(CuArray(x),N) for x in (A,Q,H,R))...)
        records = []
        for policy in (:legacy,:dual,:single,:input_register,:register,:balanced_mul,:balanced_factor)
            try
                record = prepare(covariance_step,args,policy,reference)
                record === nothing || push!(records,record)
            catch err
                println("REJECTED,covariance_step,$policy,$D: ", sprint(showerror,err))
                flush(stdout)
            end
        end
        geometry_candidates = [(threads,policy) for threads in (64,256) for policy in (:legacy,:single,:register)]
        # D=32 is shared-memory constrained; include smaller feasible balanced
        # candidates rather than concluding from their 128-thread rejection.
        if D == 32
            append!(geometry_candidates, [(32,p) for p in (:legacy,:single,:register,:balanced_mul,:balanced_factor)])
            append!(geometry_candidates, [(64,p) for p in (:balanced_mul,:balanced_factor)])
        end
        for (threads,policy) in geometry_candidates
            try
                record = prepare(covariance_step,args,policy,reference; nthreads=threads)
                record === nothing || push!(records,record)
            catch err
                println("REJECTED,covariance_step,$policy,$D,$threads: ", sprint(showerror,err))
                flush(stdout)
            end
        end
        try
            prototype = prepare_prototype(args,reference)
            prototype === nothing || push!(records,prototype)
        catch err
            println("REJECTED,covariance_step,prototype,$D: ", sprint(showerror,err))
        end
        measure!(records,rng)
        all(Array(x.data) == original for (x,original) in zip(args,(P,A,Q,H,R))) || error("Input changed during timing")
    end
    get(ENV,"BK_M6_PROBES","true") == "true" || return nothing
    # Narrow RHS makes orientation and compiler pressure visible independently of Kalman.
    D = 32
    A = .02f0 * randn(rng,Float32,D,D,N)
    for n in 1:N, i in 1:D
        A[i,i,n] = 2f0
    end
    B = randn(rng,Float32,D,2,N)
    args = (BK.BatchedCuMatrix(CuArray(A)),BK.BatchedCuMatrix(CuArray(B)))
    for f in (matmul_probe,solve_probe)
        reference = similar(B)
        for n in 1:N
            reference[:,:,n] = f(A[:,:,n],B[:,:,n])
        end
        records = []
        for orientation in (:row,:col), policy in (:single,:register)
            record = prepare(f,args,policy,reference; nthreads=64,orientation)
            record === nothing || push!(records, merge(record,(;policy=Symbol(policy,:_,orientation))))
        end
        measure!(records,rng)
        all(Array(x.data) == original for (x,original) in zip(args,(A,B))) || error("Probe input changed during timing")
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_benchmarks()
