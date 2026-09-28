@testitem "Wrapped covariance-root products preserve triangular masks" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    products(A, B, C) = (
        UpperTriangular(A) * B',
        B * UpperTriangular(A)',
        LowerTriangular(A) * C,
        B * LowerTriangular(A),
        UpperTriangular(A) * transpose(B),
        B * transpose(UpperTriangular(A)),
    )
    rng = MersenneTwister(724)
    for T in (Float32, Float64)
        # Both halves contain data: stripping the wrapper would produce the wrong result.
        As = randn(rng, T, 4, 4, 7)
        Bs = randn(rng, T, 3, 4, 7)
        Cs = randn(rng, T, 4, 2, 7)
        args = map(x -> BatchedCuMatrix(CuArray(x)), (As, Bs, Cs))
        result = fuse(products, args...)
        arrays = map(x -> Array(x.data), values(result.components))
        tol = T === Float32 ? 2e-5 : 2e-12
        for i in 1:7
            refs = products(As[:, :, i], Bs[:, :, i], Cs[:, :, i])
            for (a, r) in zip(arrays, refs)
                @test isapprox(a[:, :, i], r; rtol=tol, atol=tol)
            end
        end
        for (got, original) in zip(args, (As, Bs, Cs))
            @test Array(got.data) == original
        end
    end
end
