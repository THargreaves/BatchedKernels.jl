@testitem "Random stream and counter addresses" tags = [:cpu] begin
    using BatchedKernels, Random
    const BK = BatchedKernels
    @test_throws ArgumentError BatchedRNG(-1)
    @test_throws ArgumentError BatchedRNG(big(2)^64)
    r = BatchedRNG(123)
    @test BK._reserve_rngs([r, r], [true, true]) == fill(BK.DeviceRNG(123, 0), 2)
    @test r.counter == 1
    checkpoint = copy(r)
    @test BK._reserve_rngs([r], [true]) == BK._reserve_rngs([checkpoint], [true])
    @test checkpoint.lock !== r.lock
    Random.seed!(r, 456)
    @test r.seed == 456 && r.counter == 0
    BK._reserve_rngs([r], [false])
    @test r.counter == 0
    # Concurrent reservations are distinct even when the same RNG is shared.
    tasks = [Threads.@spawn BK._reserve_rngs([r], [true])[1].counter for _ in 1:128]
    @test sort(fetch.(tasks)) == UInt64.(0:127)
    @test r.counter == 128
    exhausted = BatchedRNG(1)
    exhausted.counter = typemax(UInt64)
    @test_throws ArgumentError BK._reserve_rngs([r, exhausted], [true, true])
    @test r.counter == 128
    # Published Philox4x32-10 zero-key/zero-counter known-answer vector.
    @test BK._random_words(BK.DeviceRNG(0, 0), UInt32(0), UInt32(0), UInt32(0)) ==
        (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
    addresses = [
        BK._random_words(BK.DeviceRNG(1, epoch), UInt32(p), UInt32(s), UInt32(e)) for
        epoch in (0, 1, 2^32), p in 0:2, s in 0:2, e in (0, 1, 1023)
    ]
    @test allunique(vec(addresses))
    # Array addresses must remain injective across both the 10-bit element and
    # 32-bit address boundaries, without allocating impractically large arrays.
    for index in UInt64[0, 1, 1023, 1024, 2^32-1, 2^32, typemax(Int)-1]
        p, s, e = BK._array_random_address(index)
        @test e < 1024 && s < 1 << 22
        @test (UInt64(p) << 32) | (UInt64(s) << 10) | UInt64(e) == index
    end
    for T in (Float32, Float64)
        @test BK._uniform(T, UInt32(0), UInt32(0)) == 0
        @test BK._uniform(T, typemax(UInt32), typemax(UInt32)) == prevfloat(one(T))
    end
end

@testitem "GPU array random fills and mixed streams" tags = [:gpu] begin
    using BatchedKernels, Random, CUDA
    const BK = BatchedKernels
    fused(r) = randn(r, Float32, 3)
    for T in (Float32, Float64), (fill!, normal) in ((rand!, false), (randn!, true))
        for dims in ((), (1,), (1025,), (3, 5, 71))
            A = CuArray{T}(undef, dims)
            r = BatchedRNG(71)
            @test fill!(r, A) === A
            @test r.counter == 1
            expected = [
                BK._random_sample(T, Val(normal), BK.DeviceRNG(71, 0),
                    BK._array_random_address(UInt64(i-1))...) for i in 1:length(A)
            ]
            if normal
                @test vec(Array(A)) ≈ expected rtol=10eps(T)
            else
                @test vec(Array(A)) == expected
            end
            B = CuArray{T}(undef, length(A))
            fill!(BatchedRNG(71), B)
            @test vec(Array(A)) == Array(B)
            # A deliberately small grid exercises repeated grid-stride writes.
            CUDA.@cuda threads=32 blocks=1 BK._fill_random_kernel!(
                B, BK.DeviceRNG(71, 0), Val(normal))
            @test vec(Array(A)) == Array(B)
        end
    end
    r = BatchedRNG(19)
    A = CuArray{Float32}(undef, 1031)
    rand!(r, A)
    checkpoint = copy(r)
    first = fuse(fused, r; batch_size=7)
    randn!(r, A)
    saved = Array(A)
    @test r.counter == 3
    @test Array(first.data) == Array(fuse(fused, checkpoint; batch_size=7).data)
    randn!(checkpoint, A)
    @test Array(A) == saved
    # Reservations also work across separate CUDA streams; reseeding after
    # submission leaves each already-enqueued snapshot unchanged.
    B = similar(A)
    Random.seed!(r, 29)
    CUDA.stream!(CUDA.CuStream()) do
        rand!(r, A)
    end
    CUDA.stream!(CUDA.CuStream()) do
        rand!(r, B)
    end
    Random.seed!(r, 29)
    CUDA.synchronize()
    ref = similar(A)
    rand!(r, ref)
    @test Array(A) == Array(ref)
    rand!(r, ref)
    @test Array(B) == Array(ref)
    @test Array(A) != Array(B)
    position = r.counter
    for f in (rand!, randn!)
        empty = CuArray{Float32}(undef, 2, 0)
        @test f(r, empty) === empty
        for T in (Float16, Int32, ComplexF32)
            @test_throws ArgumentError f(r, CuArray{T}(undef, 7))
            @test_throws ArgumentError f(r, CuArray{T}(undef, 0))
        end
        @test r.counter == position
        # Noncontiguous views are outside this API. Random's fallback must fail
        # without consuming the BatchedRNG or partially writing the destination.
        parent = CUDA.fill(Float32(-1), 5, 3)
        strided = view(parent, 1:2:5, :)
        @test_throws MethodError f(r, strided)
        @test Array(parent) == fill(Float32(-1), 5, 3)
        @test r.counter == position
    end
    r.counter = typemax(UInt64)
    @test_throws ArgumentError rand!(r, A)
    @test_throws ArgumentError randn!(r, A)
    @test r.counter == typemax(UInt64)
end

@testitem "Random tracing and inference" tags = [:cpu] begin
    using BatchedKernels, Random
    const BK = BatchedKernels
    f(r) = (rand(r, Float32), randn(r, Float32, 3), rand(r, Float32, (3, 5)))
    specs = BK.InputSpec[BK.RNGInput()]
    r = BatchedRNG(1)
    tape = BK.trace(f, specs)
    calls = findall(n -> n isa BK.CallNode && n.fn === BK._sample_random, tape.nodes)
    @test length(calls) == 3
    @test all(i -> tape.metas[i].lifecycle == BK.BATCHED, calls)
    @test [BK.node_at(tape, tape.nodes[i].args[2]).val.site for i in calls] == UInt32[0, 1, 2]
    @test Core.Compiler.return_type(f, Tuple{BK.TraceRNG}) === Tuple{
        BK.TraceScalar{Float32},BK.TraceVector{Float32,3},BK.TraceMatrix{Float32,3,5}
    }
    @test BK.input_cache_key(BK.input_spec(r)) ==
        BK.input_cache_key(BK.input_spec(BatchedRNG(99)))
    @test r.counter == 0
    phantom = BK._reconstruct_trace_arg!(BK.Tape(), BK.RNGInput())
    @test rand(phantom) isa BK.TraceScalar{Float64}
    @test randn(phantom) isa BK.TraceScalar{Float64}
    @test randn(phantom, (2, 3)) isa BK.TraceMatrix{Float64,2,3}
    @test rand(phantom, 3) isa BK.TraceVector{Float64,3}
    @test_throws ArgumentError rand(phantom, Float32, ())
    @test_throws ArgumentError randn(phantom, ())
    @test_throws ArgumentError rand(phantom, Int)
    @test_throws ArgumentError randn(phantom, Float16)
    @test_throws ArgumentError randn(phantom, Float32, 0)
    @test_throws ArgumentError rand(phantom, Float32, 33)
    @test_throws ArgumentError randn(phantom, Float32, 2, 2, 2)
    @test_throws ArgumentError BK.trace(identity, specs)
    assignment = automatic_assignment(tape; nthreads=64)
    @test BK.plan_memory(tape, assignment; D_MAX=5, T=Float32) isa BK.HybridPlannerOutput
end

@testitem "Fused random advancement and validation" tags = [:gpu] begin
    using BatchedKernels, Random, CUDA
    const BK = BatchedKernels
    sample(r) = randn(r, Float32, 3)
    with_input(r, x) = x + randn(r, eltype(x), size(x, 1))
    unused(r, x) = x + x
    two(r, s) = (rand(r, Float32), rand(s, Float32))
    r = BatchedRNG(17)
    entry = BK._ensure_compiled!(sample, (r,))
    @test r.counter == 0
    first = @inferred fuse(sample, r; batch_size=23)
    @test r.counter == 1
    checkpoint = copy(r)
    second = fuse(sample, r; batch_size=23)
    @test Array(first.data) != Array(second.data)
    @test Array(second.data) == Array(fuse(sample, checkpoint; batch_size=23).data)
    Random.seed!(r, 17)
    @test Array(first.data) == Array(fuse(sample, r; batch_size=23).data)
    @test BK._ensure_compiled!(sample, (BatchedRNG(444),)) === entry
    @test size(fuse(sample, r; batch_size=0).data) == (3, 0)
    @test r.counter == 1
    @test_throws ArgumentError fuse(sample, r)
    @test_throws ArgumentError fuse(sample, r; batch_size=-1)
    @test_throws ArgumentError fuse(sample, r; batch_size=big(2)^32)
    @test_throws ArgumentError fuse(sample, r; batch_size=1, nthreads=33)
    x = BatchedCuVector(CUDA.zeros(Float32, 3, 7))
    @test_throws ErrorException fuse(with_input, r, x; batch_size=8)
    @test r.counter == 1
    @test size(fuse(with_input, r, x).data) == (3, 7)
    fuse(unused, r, x)
    @test r.counter == 2
    # RNG behaves as a scalar broadcast operand without a batch-sized allocation.
    Random.seed!(r, 17)
    @test Array(with_input.(r, x).data) == Array(fuse(with_input, BatchedRNG(17), x).data)
    r2 = BatchedRNG(18)
    out = fuse(two, r2, r2; batch_size=31)
    @test r2.counter == 1
    a, b = values(out.components)
    @test Array(a.data) != Array(b.data)
    # Validation rejects mixed numerical storage before consuming the stream.
    mixed(r, x) = (x, rand(r, Float64))
    @test_throws ErrorException fuse(mixed, r, x)
    @test r.counter == 1
end

@testitem "Fused random storage and numerical semantics" tags = [:gpu] begin
    using BatchedKernels, Random, CUDA, LinearAlgebra
    const BK = BatchedKernels
    draws32(r) = (rand(r, Float32), randn(r, Float32, 3), rand(r, Float32, 3, 5))
    draws64(r) = (rand(r, Float64), randn(r, Float64, 3), rand(r, Float64, 3, 5))
    host(out) = map(x -> Array(x.data), values(out.components))
    for (T, f) in ((Float32, draws32), (Float64, draws64))
        for N in (1, 23, 33)
            reference = host(fuse(f, BatchedRNG(42); batch_size=N))
            for kwargs in
                ((nthreads=64,), (nthreads=256, shared_memory=:dynamic), (policy=:legacy,))
                @test host(fuse(f, BatchedRNG(42); batch_size=N, kwargs...)) == reference
            end
            # Independent host evaluation of logical counter coordinates detects
            # lane, matrix orientation, and padded-group addressing errors.
            state = BK.DeviceRNG(42, 0)
            scalar = [
                BK._random_sample(T, Val(false), state, UInt32(p-1), UInt32(0), UInt32(0))
                for p in 1:N
            ]
            vector = [
                BK._random_sample(T, Val(true), state, UInt32(p-1), UInt32(1), UInt32(i-1))
                for i in 1:3, p in 1:N
            ]
            matrix = [
                BK._random_sample(
                    T, Val(false), state, UInt32(p-1), UInt32(2), UInt32(i-1+3*(j-1))
                ) for i in 1:3, j in 1:5, p in 1:N
            ]
            @test reference[1] == scalar
            @test reference[2] ≈ vector rtol=10eps(T)
            @test reference[3] == matrix
        end
    end
    # Force each supported matrix residence/orientation on the same random node.
    matrix(r) = randn(r, Float32, 3, 5)
    tape = BK.trace(matrix, BK.InputSpec[BK.RNGInput()])
    id = only(findall(n -> n isa BK.CallNode && n.fn === BK._sample_random, tape.nodes))
    expected = Array(fuse(matrix, BatchedRNG(77); batch_size=29).data)
    for (residence, orientation) in (
        (:register, :row),
        (:register, :col),
        (:single, :row),
        (:single, :col),
        (:dual, :both),
    )
        a = Assignment(
            tape;
            residences=Dict(id=>residence),
            orientations=Dict(id=>orientation),
            variants=Dict(id=>(orientation === :col ? :random_col : :random_row)),
        )
        @test Array(fuse(matrix, BatchedRNG(77); batch_size=29, assignment=a).data) ==
            expected
    end
    # Full-warp groups, the maximum draw address, and unused larger inputs must
    # preserve logical element/particle indexing.
    edge(r, A) = rand(r, eltype(A), size(A))
    small(r, A) = randn(r, Float32, 3)
    for (m, n) in ((1, 1), (2, 31), (32, 32))
        input = SharedCuMatrix(CUDA.zeros(Float32, m, n), 3)
        got = Array(fuse(edge, BatchedRNG(91), input).data)
        expected_edge = [
            BK._random_sample(
                Float32,
                Val(false),
                BK.DeviceRNG(91, 0),
                UInt32(p-1),
                UInt32(0),
                UInt32(i-1+m*(j-1)),
            ) for i in 1:m, j in 1:n, p in 1:3
        ]
        @test got == expected_edge
        @test Array(fuse(small, BatchedRNG(91), input).data) == Array(
            fuse(small, BatchedRNG(91), SharedCuMatrix(CUDA.zeros(Float32, 3, 3), 3)).data
        )
    end
    # A scalar draw must agree across every lane of a particle's matrix group.
    scaled(r, A) = A / rand(r, Float32)
    A = SharedCuMatrix(CUDA.ones(Float32, 3, 5), 23)
    scaled_out = Array(fuse(scaled, BatchedRNG(1), A).data)
    @test all(p -> all(==(scaled_out[1, 1, p]), scaled_out[:, :, p]), 1:23)
    # Gaussian transition is ordinary scalar Julia on CPU and one fused GPU call.
    transition(r, x, A, L) = (e=randn(r, eltype(x), size(x, 1)); (A*x + L*e, e))
    xh = reshape(Float32.(1:69), 3, 23) / 50
    Ah = Float32[1 0.2 0; 0 1 0.1; 0 0 1]
    Lh = Float32[1 0 0; 0.3 2 0; 0.1 -0.2 0.5]
    out = @inferred fuse(
        transition,
        BatchedRNG(5),
        BatchedCuVector(CuArray(xh)),
        SharedCuMatrix(CuArray(Ah), 23),
        SharedCuMatrix(CuArray(Lh), 23),
    )
    y, e = host(out)
    @test y ≈ Ah*xh + Lh*e rtol=2.0f-6
    @test size(first(transition(Xoshiro(5), xh[:, 1], Ah, Lh))) == (3,)
    # Distribution smoke checks supplement deterministic addressing tests.
    normals(r) = randn(r, Float64)
    uniforms(r) = rand(r, Float64)
    z = Array(fuse(normals, BatchedRNG(321); batch_size=100_000).data)
    u = Array(fuse(uniforms, BatchedRNG(321); batch_size=100_000).data)
    @test all(isfinite, z)
    @test abs(sum(z)/length(z)) < 0.02
    @test abs(sum(abs2, z)/length(z) - 1) < 0.03
    @test all(v -> 0 <= v < 1, u)
    @test abs(sum(u)/length(u) - 0.5) < 0.005
end
