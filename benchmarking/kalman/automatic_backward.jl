# julia --project=. benchmarking/kalman/automatic_backward.jl
# Kernel-only CUDA-event timing and synchronized allocation-inclusive host timing.
include(joinpath(@__DIR__, "automatic_srkf.jl"))

function backward_benchmark_cases(::Type{T}, n, m, N; batched_A=false) where {T}
    rng = MersenneTwister(947)
    root(k) = (X = randn(rng, T, k, k); Matrix(cholesky(Symmetric(X * X' / T(k) + I)).U))
    μ = randn(rng, T, n)
    U = root(n)
    A = randn(rng, T, n, n) / sqrt(T(n))
    b = randn(rng, T, n)
    UQ = root(n)
    H = randn(rng, T, m, n) / sqrt(T(n))
    c = randn(rng, T, m)
    UR = root(m)
    y = randn(rng, T, m)
    B, r, logscale = sqrt_backward_initialise(H, c, UR, y)
    batched(x::Number) = BatchedCuScalar(CuArray(fill(x, N)))
    batched(x::AbstractVector) = BatchedCuVector(CuArray(repeat(x, 1, N)))
    batched(x::AbstractMatrix) = BatchedCuMatrix(CuArray(repeat(x, 1, 1, N)))
    shared(x::AbstractVector) = SharedCuVector(CuArray(x), N)
    shared(x::AbstractMatrix) = SharedCuMatrix(CuArray(x), N)
    a = batched_A ? batched(A) : shared(A)
    initargs = (shared(H), shared(c), shared(UR), batched(y))
    stepargs = (
        batched(B),
        batched(r),
        batched(logscale),
        a,
        shared(b),
        shared(UQ),
        shared(H),
        shared(c),
        shared(UR),
        shared(y),
    )
    weightargs = (
        batched(μ),
        batched(U),
        a,
        shared(b),
        shared(UQ),
        shared(B),
        shared(r),
        batched(T(-2)),
        batched(T(-1)),
    )
    covargs = (
        weightargs[1], batched(U'U), a, shared(b), shared(UQ'UQ), weightargs[6:end]...
    )
    return (
        (:initialise, sqrt_backward_initialise, initargs, (B, r, logscale)),
        (
            :backward_step,
            sqrt_backward_step,
            stepargs,
            sqrt_backward_step(B, r, logscale, A, b, UQ, H, c, UR, y),
        ),
        (
            :sr_weight,
            sqrt_backward_weight,
            weightargs,
            sqrt_backward_weight(μ, U, A, b, UQ, B, r, T(-2), T(-1)),
        ),
        (
            :cov_weight,
            kalman_backward_weight,
            covargs,
            kalman_backward_weight(μ, U'U, A, b, UQ'UQ, B, r, T(-2), T(-1)),
        ),
    )
end

function measure_backward(stage, f, args, reference, T, n, m, N; nthreads=128, samples=30)
    first_seconds = @elapsed begin
        out = fuse(f, args...; nthreads)
        CUDA.synchronize()
    end
    leaves = out isa BatchedStruct ? values(out.components) : (out,)
    ref = reference isa Tuple ? reference : (reference,)
    tol = T === Float32 ? 1e-4 : 3e-12
    for (leaf, r) in zip(leaves, ref)
        cpu = Array(leaf.data)
        first = if ndims(cpu) == 3
            cpu[:, :, 1]
        elseif ndims(cpu) == 2
            cpu[:, 1]
        else
            cpu[1]
        end
        @assert isapprox(first, r; rtol=tol, atol=tol)
    end
    entry = BK._ensure_compiled!(f, args; nthreads)
    batched, shared = Any[], Any[]
    nref = Ref{Union{Nothing,Int}}(nothing)
    for a in args
        BK._collect_runtime_inputs!(batched, shared, nref, a)
    end
    launch_args = (map(x -> x.data, leaves)..., batched..., shared..., Int32(N))
    blocks = cld(N, (nthreads ÷ 32) * (32 ÷ entry.D_MAX))
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
        CUDA.@sync fuse(f, args...; nthreads)
    end
    host_us = median([
        1e6 * @elapsed(CUDA.@sync fuse(f, args...; nthreads)) for _ in 1:samples
    ])
    memory = CUDA.memory(kernel)
    println(
        join(
            (
                T,
                n,
                m,
                N,
                stage,
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

function backward_main()
    println("# GPU: ", CUDA.name(CUDA.device()), "; Julia: ", VERSION)
    println(
        "type,n,m,batch,stage,threads,registers,local_bytes,shared_bytes,first_seconds,kernel_us,host_us",
    )
    for (T, n, m) in ((Float32, 3, 2), (Float32, 8, 4), (Float32, 16, 6), (Float64, 9, 4))
        N = 8192
        for (stage, f, args, ref) in
            backward_benchmark_cases(T, n, m, N; batched_A=T === Float64)
            measure_backward(stage, f, args, ref, T, n, m, N)
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    backward_main()
end
