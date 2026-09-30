@testitem "Composite input: BatchedStruct walked as a composite trace input" begin
    # Wrap two batched matrices in a scalar struct (PairMat) and broadcast a
    # function `x.A * x.B` over the resulting `BatchedStruct`. The fuser must
    # introspect the BatchedStruct's NamedTuple, derive the per-leaf trace
    # types (substituting TraceMatrix for the batched components), and walk
    # `getfield` reads on the trace value to build the tape. Verifies both
    # correctness against a CPU reference and `@inferred` type stability.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    struct PairMat{A,B}
        A::A
        B::B
    end

    N = 2^9 + 1
    T = Float32
    D = 4

    CUDA.seed!(1234)
    A_cpu = randn(T, D, D, N)
    B_cpu = randn(T, D, D, N)

    A_gpu = BatchedCuMatrix(CuArray(A_cpu))
    B_gpu = BatchedCuMatrix(CuArray(B_cpu))
    comps = (A=A_gpu, B=B_gpu)
    # Carry the scalar struct type with abstract field types so the trace pass
    # can substitute TraceMatrix in for the batched components without a
    # type-parameter mismatch.
    PMT = PairMat{AbstractMatrix{T},AbstractMatrix{T}}
    pm = BatchedStruct{PMT,typeof(comps)}(comps, N)

    f(x) = x.A * x.B
    g(x) = f.(x)

    result = @inferred g(pm)
    @test result isa BatchedCuMatrix{T,D,D}
    got = Array(result.data)

    ref = Array{T}(undef, D, D, N)
    for n in 1:N
        ref[:, :, n] = A_cpu[:, :, n] * B_cpu[:, :, n]
    end
    @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
end

@testitem "Fused composite outputs are labelled like hand-built batches" begin
    using BatchedKernels
    using CUDA

    struct Belief{M,C}
        mean::M
        covariance::C
    end
    doubled(b) = Belief(b.mean + b.mean, b.covariance + b.covariance)
    sum_difference(a, b) = (a + b, a - b)

    N = 33
    μ = BatchedCuVector(CUDA.rand(Float32, 3, N))
    Σ = BatchedCuMatrix(CUDA.rand(Float32, 3, 3, N))
    built = BatchedStruct(Belief, (; mean=μ, covariance=Σ))

    fused = @inferred fuse(doubled, built)
    @test eltype(fused) === eltype(built)
    @test eltype(@inferred fuse(doubled, fused)) === eltype(built)
    @test eltype(@inferred fuse(sum_difference, μ, μ)) === Tuple{eltype(μ),eltype(μ)}

    # A view-backed input still yields leaves in freshly allocated storage.
    strided = BatchedCuMatrix(view(CUDA.rand(Float32, 3, 3, 2N), :, :, 1:2:(2N)))
    @test eltype(fuse(doubled, BatchedStruct(Belief, (; mean=μ, covariance=strided)))) ===
        eltype(built)
end
