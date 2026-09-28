@testitem "Composite batch construction without scalar indexing" tags=[:cpu] begin
    using BatchedKernels, LinearAlgebra, StaticArrays

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
    struct NestedParameter{T,N}
        value::SVector{N,T}
    end
    struct MatchingFields{T}
        a::T
        b::T
    end
    struct NoIndex{T} <: AbstractVector{T}
        n::Int
    end
    Base.size(x::NoIndex) = (x.n,)
    Base.getindex(::NoIndex, ::Int) = error("constructor must not index a batch")

    n = 3
    backing = reshape(collect(Float32, 1:24), 2, 2, 6)
    matrices = BatchedCuMatrix(view(backing, :, :, 1:2:6))
    means = BatchedCuVector(reshape(collect(Float32, 1:6), 2, n))
    belief = @inferred BatchedStruct(Belief, (; covariance=matrices, mean=means))
    @test keys(belief.components) == (:mean, :covariance)
    @test eltype(belief) === Belief{eltype(means),eltype(matrices)}
    @test belief.components.covariance.data === matrices.data
    @test belief[2].covariance == backing[:, :, 3]

    state = @inferred BatchedStruct(State, (; x=means, z=belief))
    ancestry = BatchedCuScalar(Int64[2^40, 2^40+1, 2^40+2])
    weights = BatchedCuScalar(zeros(Float32,n))
    weighted = @inferred BatchedStruct(Weighted, (; state, log_w=weights, ancestor=ancestry))
    @test eltype(weighted) === Weighted{eltype(state),Float32,Int64}
    @test weighted.components.ancestor === ancestry
    @test weighted[2].ancestor == Int64(2)^40+1

    # Concrete logical types may have abstract fields, as generated outputs do.
    abstract_belief = BatchedStruct(Belief{AbstractVector{Float32},AbstractMatrix{Float32}},
        (; mean=means, covariance=matrices))
    @test eltype(abstract_belief) === Belief{AbstractVector{Float32},AbstractMatrix{Float32}}
    @test abstract_belief[1].mean == means[1]
    @test_throws ArgumentError BatchedStruct(NestedParameter,
        (; value=NoIndex{SVector{2,Float32}}(n)))
    nested = BatchedStruct(NestedParameter{Float32,2},
        (; value=NoIndex{SVector{2,Float32}}(n)))
    @test eltype(nested) === NestedParameter{Float32,2}
    @test length(nested) == n
    opaque = BatchedStruct(Weighted, (; state=NoIndex{eltype(state)}(n),
        log_w=NoIndex{Float32}(n), ancestor=NoIndex{Int64}(n)))
    @test eltype(opaque) === eltype(weighted)

    matched = @inferred BatchedStruct(MatchingFields,
        (; a=NoIndex{AbstractVector{Float32}}(n), b=NoIndex{AbstractVector{Float32}}(n)))
    @test eltype(matched) === MatchingFields{AbstractVector{Float32}}
    @test_throws ArgumentError BatchedStruct(MatchingFields,
        (; a=NoIndex{Float32}(n), b=NoIndex{Float64}(n)))
    @test_throws ArgumentError BatchedStruct(Belief, (; mean=means, wrong=matrices))
    @test_throws DimensionMismatch BatchedStruct(Belief,
        (; mean=means, covariance=BatchedCuMatrix(zeros(Float32,2,2,n+1))))
    @test_throws ArgumentError BatchedStruct(Weighted,
        (; state, log_w=weights, ancestor=BatchedCuScalar(zeros(Float32,n))))
    @test_throws ArgumentError BatchedStruct(Belief{Vector{Float64},Matrix{Float64}},
        (; mean=means, covariance=matrices))
end
