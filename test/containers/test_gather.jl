@testitem "Batch gathering of nested composites" tags=[:cpu] begin
    using BatchedKernels

    struct Belief{M,C}
        mean::M
        covariance::C
    end
    struct State{X,Z}
        x::X
        z::Z
    end
    struct Weighted{S,W,A<:Integer}
        state::S
        log_w::W
        ancestor::A
    end

    n = 5
    # Duplicates, reordering and both boundaries; the batch length also changes.
    idxs = [n; 1:(n - 2); n; 2]
    x_host = reshape(collect(Float32, 1:(2n)), 2, n)
    μ_host = reshape(collect(Float32, 1:(2n)) ./ 10, 2, n)
    backing_host = reshape(collect(Float32, 1:(6n)), 3, 2, n)
    backing = copy(backing_host)
    # Beyond Float32's exact integer range.
    ancestry = Int64(2)^40 .+ collect(Int64, 1:n)

    x = BatchedCuVector(copy(x_host))
    μ = BatchedCuVector(copy(μ_host))
    # A sub-block view gathers into contiguous storage with a different view type.
    Σ = BatchedCuMatrix(view(backing, 1:2, :, :))
    z = BatchedStruct(Belief, (; mean=μ, covariance=Σ))
    state = BatchedStruct(State, (; x, z))
    log_w = BatchedCuScalar(-collect(Float32, 1:n))
    particles = BatchedStruct(
        Weighted, (; state, log_w, ancestor=BatchedCuScalar(copy(ancestry)))
    )

    gathered = @inferred particles[idxs]
    c = gathered.components
    @test length(gathered) == length(idxs)
    @test c.state.components.x.data == x_host[:, idxs]
    @test c.state.components.z.components.mean.data == μ_host[:, idxs]
    @test c.state.components.z.components.covariance.data == backing_host[1:2, :, idxs]
    @test c.log_w.data == log_w.data[idxs]
    @test c.ancestor.data isa Vector{Int64}
    @test c.ancestor.data == ancestry[idxs]

    contiguous = eltype(BatchedCuMatrix(zeros(Float32, 2, 2, 0)))
    @test eltype(Σ) !== contiguous
    @test eltype(gathered) ===
        Weighted{State{eltype(x),Belief{eltype(μ),contiguous}},Float32,Int64}
    @test gathered[1].state.z.covariance == backing_host[1:2, :, n]
    @test gathered[1].ancestor === ancestry[n]

    # Writing to the gathered population must not alter its source.
    c.state.components.x.data .= -1
    c.state.components.z.components.covariance.data .= -1
    c.ancestor.data .= 0
    @test x.data == x_host
    @test backing == backing_host
    @test particles.components.ancestor.data == ancestry

    L = SharedCuMatrix(Float32[1 0; 1 1], n)
    shared = BatchedStruct(Belief, (; mean=μ, covariance=L))[idxs]
    @test shared.components.covariance.data === L.data
    @test length(shared.components.covariance) == length(idxs)
    literal = SharedValue('L', n)[idxs]
    @test literal.value === 'L'
    @test length(literal) == length(idxs)

    # Ordinary vector components are gathered by their own indexing.
    plain = BatchedStruct(Belief, (; mean=collect(1:n), covariance=L))[idxs]
    @test plain.components.mean == idxs
    @test eltype(plain) === Belief{Int,eltype(L)}

    # Declared field types that the gathered leaves still satisfy are retained.
    Abstract = Belief{AbstractVector{Float32},AbstractMatrix{Float32}}
    @test eltype(BatchedStruct(Abstract, (; mean=μ, covariance=Σ))[idxs]) === Abstract

    @test length(particles[Int[]]) == 0
    @test_throws BoundsError particles[[0]]
    @test_throws BoundsError particles[[n + 1]]
    @test_throws ArgumentError particles[trues(n)]
end

@testitem "Batch gathering with device indices" begin
    using BatchedKernels, CUDA

    struct Belief{M,C}
        mean::M
        covariance::C
    end
    struct State{X,Z}
        x::X
        z::Z
    end
    struct Weighted{S,W,A<:Integer}
        state::S
        log_w::W
        ancestor::A
    end

    CUDA.allowscalar(false)
    n = 33
    idxs = [n; 1:(n - 2); n]
    device_idxs = CuArray(Int32.(idxs))
    x_host = reshape(collect(Float32, 1:(3n)), 3, n)
    Σ_host = reshape(collect(Float32, 1:(9n)), 3, 3, n)
    ancestry = Int64(2)^40 .+ collect(Int64, 1:n)

    x = BatchedCuVector(CuArray(x_host))
    μ = SharedCuVector(CUDA.ones(Float32, 3), n)
    Σ = BatchedCuMatrix(CuArray(Σ_host))
    z = BatchedStruct(Belief, (; mean=μ, covariance=Σ))
    state = BatchedStruct(State, (; x, z))
    particles = BatchedStruct(
        Weighted,
        (;
            state,
            log_w=BatchedCuScalar(CUDA.zeros(Float32, n)),
            ancestor=BatchedCuScalar(CuArray(ancestry)),
        ),
    )

    gathered = particles[device_idxs]
    c = gathered.components
    @test eltype(gathered) === eltype(particles)
    @test Array(c.state.components.x.data) == x_host[:, idxs]
    @test Array(c.state.components.z.components.covariance.data) == Σ_host[:, :, idxs]
    @test c.ancestor.data isa CuVector{Int64}
    @test Array(c.ancestor.data) == ancestry[idxs]
    @test c.state.components.z.components.mean.data === μ.data
    @test length(c.state.components.z.components.mean) == n

    raw = BatchedStruct(Belief, (; mean=CuArray(ancestry), covariance=Σ))[device_idxs]
    @test Array(raw.components.mean) == ancestry[idxs]
    @test eltype(raw) === Belief{Int64,eltype(Σ)}

    c.state.components.z.components.covariance.data .= -1
    @test Array(Σ.data) == Σ_host

    @test_throws BoundsError particles[CuArray(Int32[0])]

    # Gathering a fused tuple output keeps its element type.
    sums = fuse((a, b) -> (a + b, a - b), x, x)
    gathered_sums = sums[device_idxs]
    @test eltype(gathered_sums) === eltype(sums)
    @test Array(first(gathered_sums.components).data) == 2 .* x_host[:, idxs]
end
