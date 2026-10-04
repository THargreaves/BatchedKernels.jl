@testitem "Runtime shared scalar contract and structural literals" tags = [:cpu] begin
    using BatchedKernels
    const BK = BatchedKernels
    struct Config{T,M}
        scale::T
        mode::M
    end
    struct FixedScale
        scale::Float32
    end
    struct RealScale{T<:Real}
        scale::T
    end

    runtime = shared(2.0f0, 5)
    @test runtime isa SharedScalar{Float32}
    @test runtime[3] === 2.0f0
    @test runtime[[5, 1, 1]] isa SharedScalar{Float32}
    @test length(runtime[[5, 1, 1]]) == 3
    @test BK.input_cache_key(BK.input_spec(runtime)) ==
        BK.input_cache_key(BK.input_spec(shared(7.0f0, 5)))
    @test BK.input_cache_key(BK.input_spec(literal(2.0f0, 5))) !=
        BK.input_cache_key(BK.input_spec(literal(7.0f0, 5)))

    configured = BatchedStruct(Config, (; scale=runtime, mode=literal(true, 5)))
    choose(c) = c.mode ? c.scale : -c.scale
    tape = BK.trace(choose, BK.InputSpec[BK.input_spec(configured)])
    @test length(tape.inputs) == 1
    @test BK.trace_element_type(typeof(configured)) === Config{BK.TraceScalar{Float32},Bool}
    @test_throws ArgumentError shared(Config(2.0f0, true), 5)
    @test_throws ArgumentError shared(FixedScale(2.0f0), 5)
    @test_throws ArgumentError shared(RealScale(2.0f0), 5)
    for value in (1, true, Float16(2), 2.0f0 + 1.0f0im, big(2))
        @test_throws ArgumentError shared(value, 5)
    end

    branch(s) = s > 0.0f0 ? s : -s
    nanbranch(s) = isnan(s) ? zero(s) : s
    selfcompare(s) = s == s ? s : zero(s)
    for f in (branch, nanbranch, selfcompare)
        @test_throws ArgumentError BK.trace(f, BK.InputSpec[BK.input_spec(runtime)])
    end
end

@testitem "Changing shared scalars reuse kernels and preserve composite values" tags = [
    :gpu
] begin
    using BatchedKernels, CUDA
    const BK = BatchedKernels
    CUDA.allowscalar(false)
    struct Parameters{T}
        scale::T
    end
    struct Model{P,T}
        parameters::P
        offset::T
    end
    score(v, model) = model.parameters.scale * sum(abs2, v) + model.offset
    unused(v, model) = sum(abs2, v) + zero(model.offset)
    extract(model) = (model.parameters.scale, model.offset)
    firstvalue(a, b) = a
    N = 19
    host = reshape(Float32.(1:(3N)) ./ 10.0f0, 3, N)
    vectors = BatchedCuVector(CuArray(host))
    norms = vec(sum(abs2, host; dims=1))

    # A partial final block and both shared-memory modes exercise argument binding
    # and scalar output staging without a Cartesian sweep over policies.
    for options in ((;), (; policy=:legacy), (; shared_memory=:dynamic))
        model = shared(Model(Parameters(2.0f0), 3.0f0), N)
        first = fuse(score, vectors, model; options...)
        @test Array(first.data) ≈ 2.0f0 .* norms .+ 3.0f0
        entries = length(BK.KERNEL_CACHE)
        changed = fuse(
            score, vectors, shared(Model(Parameters(4.0f0), -2.0f0), N); options...
        )
        @test Array(changed.data) ≈ 4.0f0 .* norms .- 2.0f0
        @test length(BK.KERNEL_CACHE) == entries
    end

    original = shared(Model(Parameters(2.0f0), 3.0f0), N)
    ignored = fuse(unused, vectors, original)
    entries = length(BK.KERNEL_CACHE)
    updated = @inferred fuse(unused, vectors, shared(Model(Parameters(8.0f0), 9.0f0), N))
    @test Array(ignored.data) ≈ Array(updated.data) ≈ norms
    @test length(BK.KERNEL_CACHE) == entries
    fields = @inferred fuse(extract, original)
    @test Array(fields.components._1.data) == fill(2.0f0, N)
    @test Array(fields.components._2.data) == fill(3.0f0, N)
    @test Array(fuse(+, fields.components._1, shared(5.0f0, N)).data) == fill(7.0f0, N)
    @test_throws "Inconsistent eltype across inputs" fuse(
        firstvalue, shared(1.0f0, N), shared(1.0, N)
    )
end

@testitem "Shared-only Float64 arithmetic and explicit literal branches" tags = [:gpu] begin
    using BatchedKernels, CUDA
    const BK = BatchedKernels
    arithmetic(a, b) = (a * b + a) / b
    choose(scale, mode) = mode ? scale : -scale
    N = 7
    result = @inferred fuse(arithmetic, shared(2.0, N), shared(4.0, N))
    @test eltype(result) === Float64
    @test Array(result.data) == fill(2.5, N)
    entries = length(BK.KERNEL_CACHE)
    @test Array(fuse(arithmetic, shared(3.0, N), shared(2.0, N)).data) == fill(4.5, N)
    @test length(BK.KERNEL_CACHE) == entries
    @test Array(fuse(identity, shared(6.0, N)).data) == fill(6.0, N)

    @test Array(fuse(choose, shared(2.0f0, N), literal(true, N)).data) == fill(2.0f0, N)
    entries = length(BK.KERNEL_CACHE)
    @test Array(fuse(choose, shared(2.0f0, N), literal(false, N)).data) == fill(-2.0f0, N)
    @test length(BK.KERNEL_CACHE) == entries + 1
    @test Array(fuse(choose, shared(4.0f0, N), literal(false, N)).data) == fill(-4.0f0, N)
    @test length(BK.KERNEL_CACHE) == entries + 1
end
