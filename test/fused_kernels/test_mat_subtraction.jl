@testitem "Matrix subtraction trace validation" tags = [:cpu] begin
    using BatchedKernels
    const BK = BatchedKernels
    const M = BK.TraceMatrix{Float32,3,5}
    specs(T) = BK.InputSpec[BK.LeafInput(M, BK.BATCHED), BK.LeafInput(T, BK.BATCHED)]
    tape = BK.trace(-, specs(M))
    calls = [i for (i, n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === (-)]
    @test length(calls) == 1
    @test tape.metas[only(calls)].type === M
    @test_throws ArgumentError M(BK.Tape(), BK.NodeRef(1)) - M(BK.Tape(), BK.NodeRef(1))
    @test_throws DimensionMismatch BK.trace(-, specs(BK.TraceMatrix{Float32,5,3}))
    @test_throws DimensionMismatch BK.trace(-, specs(BK.TraceMatrix{Float32,3,4}))
    @test_throws ErrorException BK.trace(-, specs(BK.TraceMatrix{Float64,3,5}))
end

@testitem "Fused rectangular matrix subtraction" begin
    using BatchedKernels, CUDA, Random

    function composed(A, B, C)
        difference = A - B
        return difference * C + A, difference
    end
    rng = Xoshiro(794)
    for (T, m, n, batches) in ((Float32, 3, 5, 23), (Float64, 5, 3, 1))
        a, b, c = randn(rng, T, m, n, batches),
        randn(rng, T, m, n, batches),
        randn(rng, T, n, n, batches)
        A, B, C = BatchedCuMatrix(CuArray(a)),
        BatchedCuMatrix(CuArray(b)),
        BatchedCuMatrix(CuArray(c))
        tolerance = T === Float32 ? 5.0f-5 : 2e-12
        expected = cat(
            (composed(a[:, :, i], b[:, :, i], c[:, :, i])[1] for i in 1:batches)...; dims=3
        )
        for policy in (:legacy, :auto)
            difference = @inferred fuse(-, A, B; policy)
            @test Array(difference.data) == a - b
            result = @inferred fuse(composed, A, B, C; policy)
            product, intermediate = values(result.components)
            @test Array(product.data) ≈ expected rtol=tolerance atol=tolerance
            @test Array(intermediate.data) == a - b
            @test Array(A.data) == a
            @test Array(B.data) == b
            @test Array(C.data) == c
            shared = SharedCuMatrix(CuArray(b[:, :, 1]), batches)
            @test Array(fuse(-, A, shared; policy).data) == a .- b[:, :, 1]
            @test Array(fuse(-, shared, A; policy).data) == b[:, :, 1] .- a
            @test Array(shared.data) == b[:, :, 1]
        end
    end
end
