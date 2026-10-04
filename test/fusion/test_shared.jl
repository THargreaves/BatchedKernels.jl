@testitem "Explicit shared scalar and unsupported inputs" tags = [:cpu] begin
    using BatchedKernels

    struct Label{T,S}
        count::T
        scale::S
    end
    mutable struct MutableLabel
        count::Int
    end
    struct EmptyLabel end

    value = Label(3.0f0, 2.0f0)
    batch = shared(value, 7)
    @test length(batch) == 7
    @test eltype(batch) === typeof(value)
    @test batch.components.count isa SharedScalar{Float32}
    @test batch.components.scale.value === 2.0f0
    @test batch[2] === value
    @test length(shared(value, 0)) == 0
    @test shared('U', 3).value === 'U'
    @test shared(nothing, 3).value === nothing
    @test_throws ArgumentError shared(value, -1)
    @test_throws ArgumentError shared(value, big(typemax(Int)) + 1)
    @test_throws ArgumentError shared(zeros(2), 3)
    @test_throws ArgumentError shared((value,), 3)
    @test_throws ArgumentError shared((; value), 3)
    @test_throws ArgumentError shared(MutableLabel(3), 3)
    @test_throws ArgumentError shared(EmptyLabel(), 3)
    @test_throws ArgumentError shared(big(3), 3)
    @test_throws ArgumentError shared(batch, 3)
end

@testitem "Explicit shared composite storage and fusion" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra
    const BK = BatchedKernels
    CUDA.allowscalar(false)

    struct GaussianLike{M,F}
        mean::M
        factor::F
    end
    struct ScaledGaussian{G,T,K}
        gaussian::G
        scale::T
        label::K
    end
    # Models may wrap a rectangular covariance factor in an AbstractMatrix.
    # Sharing preserves that representation instead of materializing covariance.
    struct FactorWrapper{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
        factor::M
    end
    struct FixedVector
        value::CuVector{Float32}
    end

    mean = CuArray(Float32[1, 2, 3])
    factor = CuArray(Float32[2 1 0; 9 3 1; 8 7 4])
    atom = ScaledGaussian(GaussianLike(mean, UpperTriangular(factor)), 0.5f0, 'k')
    batch = shared(atom, 5)
    @test batch isa BatchedStruct
    @test eltype(batch) === typeof(atom)
    @test batch.components.gaussian.components.mean.data === mean
    triangular = batch.components.gaussian.components.factor
    @test triangular isa BatchedStruct
    @test triangular.components.data.data === factor
    @test triangular[1] isa UpperTriangular
    @test parent(triangular[1]) === factor
    @test batch.components.label.value === 'k'
    @test eltype(shared(atom, 0)) === typeof(atom)

    vectors = Float32[1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7]
    x = BatchedCuVector(CuArray(vectors))
    function draw(a, z)
        residual = a.gaussian.mean + (a.gaussian.factor' \ z)
        return a.scale * dot(residual, residual)
    end
    actual = fuse(draw, batch, x)
    residuals = Array(mean) .+ (UpperTriangular(Array(factor))' \ vectors)
    expected = vec(0.5f0 .* sum(abs2, residuals; dims=1))
    @test Array(actual.data) ≈ expected
    # Borrowed storage stays live, and does not get captured in the trace cache.
    fill!(mean, 4.0f0)
    residuals = 4.0f0 .+ (UpperTriangular(Array(factor))' \ vectors)
    @test Array(fuse(draw, batch, x).data) ≈ vec(0.5f0 .* sum(abs2, residuals; dims=1))

    rectangular = CuArray(Float32[1 0; 0 2; 1 1])
    wrapped = FactorWrapper{Float32,typeof(rectangular)}(rectangular)
    covariance = shared(GaussianLike(mean, wrapped), 5)
    @test covariance.components.factor.components.factor.data === rectangular
    z = BatchedCuVector(CUDA.ones(Float32, 2, 5))
    sample(a, z) = a.mean + a.factor.factor * z
    @test Array(fuse(sample, covariance, z).data) ≈
        Array(mean) .+ Array(rectangular) * ones(Float32, 2, 5)

    for wrapper in (
        adjoint(factor),
        transpose(factor),
        LowerTriangular(factor),
        UnitUpperTriangular(factor),
        UnitLowerTriangular(factor),
        Symmetric(factor, :U),
    )
        wrapped_batch = shared(wrapper, 5)
        @test eltype(wrapped_batch) === typeof(wrapper)
        @test parent(wrapped_batch[1]) === factor
        # The wrapper survives phantom reconstruction; supported arithmetic is
        # still governed by the existing tracer overloads.
        @test BK.trace_element_type(typeof(wrapped_batch)) <: AbstractMatrix{Float32}
    end
    # Cholesky.info is an integer structural field, not a runtime numeric input.
    chol = shared(Cholesky(factor, 'U', 0), 5)
    @test chol.components.factors.data === factor
    @test chol.components.info isa SharedValue{Int}
    @test Array(fuse(logdet, chol).data) ≈ fill(2.0f0 * log(24.0f0), 5)
    @test_throws ArgumentError shared(Hermitian(factor), 5)
    @test_throws ArgumentError shared(FixedVector(mean), 5)
    @test_throws ArgumentError shared(CUDA.zeros(Float32, 2, 2, 2), 5)
    @test_throws ArgumentError shared(x, 5)
end
