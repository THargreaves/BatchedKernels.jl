# Run from the repository root: julia --project=. benchmarking/kalman/automatic_joseph.jl
# This measures the complete mean/covariance/likelihood graph. It is not directly
# comparable with the older covariance-only M6 measurements.
using BatchedKernels, CUDA, LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "..", "examples", "kalman.jl"))
const BK = BatchedKernels

function joseph_benchmark_args(::Type{T}, d, m, N; batched_A=false) where {T}
    rng = MersenneTwister(917)
    spd(k) = (x = randn(rng, T, k, k); x * x' / T(k) + Matrix{T}(I, k, k))
    scalar = (
        randn(rng, T, d),
        spd(d),
        randn(rng, T, d, d) / sqrt(T(d)),
        randn(rng, T, d),
        spd(d),
        randn(rng, T, m, d) / sqrt(T(d)),
        randn(rng, T, m),
        spd(m),
        randn(rng, T, m),
    )
    args = Tuple(
        if i in (1, 2) || (batched_A && i == 3)
            if ndims(x) == 2
                BatchedCuMatrix(CuArray(repeat(x, 1, 1, N)))
            else
                BatchedCuVector(CuArray(repeat(x, 1, N)))
            end
        else
            ndims(x) == 2 ? SharedCuMatrix(CuArray(x), N) : SharedCuVector(CuArray(x), N)
        end for (i, x) in enumerate(scalar)
    )
    return args, joseph_kalman_step(scalar...)
end

function measure_joseph(T, d, m, N, policy, nthreads; batched_A=false, samples=30)
    args, reference = joseph_benchmark_args(T, d, m, N; batched_A)
    options = if policy === :shared
        tape = BK.trace(joseph_kalman_step, BK.InputSpec[BK.input_spec(x) for x in args])
        automatic = automatic_assignment(tape; nthreads)
        residences = Dict(
            i => :dual for (i, r) in automatic.residences if
            r === :register && !(tape.nodes[i] isa BK.NewNode)
        )
        assignment = Assignment(tape; residences, variants=automatic.variants, nthreads)
        plan = BK.plan_memory(tape, assignment; D_MAX=max(d, m), T)
        limit = CUDA.attribute(
            CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
        )
        shared_memory = plan.shared_bytes > limit ? :dynamic : :static
        (; assignment, nthreads, shared_memory)
    else
        (; policy, nthreads)
    end
    first_seconds = @elapsed begin
        out = fuse(joseph_kalman_step, args...; options...)
        CUDA.synchronize()
    end
    leaves = values(getfield(out, :components))
    cpu = map(x -> Array(x.data), leaves)
    tolerance = T === Float32 ? 5e-5 : 2e-12
    @assert isapprox(cpu[1][:, 1], reference[1]; rtol=tolerance, atol=tolerance)
    @assert isapprox(cpu[2][:, :, 1], reference[2]; rtol=tolerance, atol=tolerance)
    @assert isapprox(cpu[3][1], reference[3]; rtol=tolerance, atol=tolerance)
    entry = BK._ensure_compiled!(joseph_kalman_step, args; options...)
    batched, shared = Any[], Any[]
    nref = Ref{Union{Nothing,Int}}(nothing)
    for a in args
        BK._collect_runtime_inputs!(batched, shared, nref, a)
    end
    launch_args = (map(x -> x.data, leaves)..., batched..., shared..., Int32(N))
    blocks = cld(N, (nthreads ÷ 32) * (32 ÷ entry.D_MAX))
    # Generated functions are newer than this host function's world.
    kernel, kernel_us = Base.invokelatest() do
        fn = entry.fn
        kernel = @cuda launch = false fn(launch_args...)
        shmem = BK._configure_dynamic_shared!(kernel, entry)
        for _ in 1:3
            CUDA.@sync kernel(launch_args...; threads=nthreads, blocks, shmem)
        end
        times = [
            1e6 * CUDA.@elapsed(kernel(launch_args...; threads=nthreads, blocks, shmem)) for
            _ in 1:samples
        ]
        kernel, median(times)
    end
    for _ in 1:3
        CUDA.@sync fuse(joseph_kalman_step, args...; options...)
    end
    host_us = median([
        1e6 * @elapsed(CUDA.@sync fuse(joseph_kalman_step, args...; options...)) for
        _ in 1:samples
    ])
    memory = CUDA.memory(kernel)
    println(
        join(
            (
                T,
                d,
                m,
                N,
                batched_A ? "batched_A" : "common",
                policy,
                nthreads,
                CUDA.registers(kernel),
                memory.local,
                memory.shared + entry.sig.dynamic_shared_bytes,
                first_seconds,
                kernel_us,
                host_us,
            ),
            ',',
        ),
    )
    return flush(stdout)
end

function main()
    println("# GPU: ", CUDA.name(CUDA.device()), "; Julia: ", VERSION)
    println(
        "type,state_dim,obs_dim,batch,lifecycle,policy,threads,registers,local_bytes,shared_bytes,first_call_seconds,kernel_us,cached_call_us",
    )
    for (T, d, m, batched_A) in (
        (Float32, 3, 2, false),
        (Float32, 8, 4, false),
        (Float64, 9, 4, true),
        (Float32, 16, 6, false),
    )
        for policy in (:auto, :legacy, :shared)
            if T === Float64 && policy === :legacy
                # Legacy masked Cholesky calls shfl_idx_f32; it cannot compile Float64.
                println("# Float64 legacy unavailable: masked Cholesky requires Float32")
                continue
            end
            measure_joseph(T, d, m, 8192, policy, 128; batched_A)
        end
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
