@testitem "Conventional Kalman arithmetic" begin
    using BatchedKernels, CUDA, LinearAlgebra, Random

    function conventional(A, v, w, s)
        S = (A + A') / 2
        Sc = cholesky(Symmetric(S))
        Si = Sc \ one(S)
        return (S, Si, dot(v, w) / 2, dot(v, Si * v) / s, 2 / dot(w, w), A / s)
    end
    identity_matrix(A) = one(A)

    rng = Xoshiro(491)
    n = 5 # A partial batch, with a non-power-of-two matrix group.
    for T in (Float32, Float64)
        raw = randn(rng, T, 3, 3, n)
        matrices = similar(raw)
        for i in 1:n
            # Retain a nonsymmetric component to exercise A + A' explicitly.
            matrices[:, :, i] =
                raw[:, :, i] * raw[:, :, i]' +
                T(2) * I +
                T(0.1) * (raw[:, :, i] - raw[:, :, i]')
        end
        v, w = randn(rng, T, 3, n), randn(rng, T, 3, n)
        scales = T[2, 3, -2, 0.5, 4]
        args = (
            BatchedCuMatrix(CuArray(matrices)),
            BatchedCuVector(CuArray(v)),
            BatchedCuVector(CuArray(w)),
            BatchedCuScalar(CuArray(scales)),
        )
        expected = [
            conventional(matrices[:, :, i], v[:, i], w[:, i], scales[i]) for i in 1:n
        ]
        tolerance = T === Float32 ? 3.0f-5 : 2e-12
        # CUDA 5 / Julia 1.12 retains an unsupported scalar-input bounds-error
        # path in larger Float64 legacy graphs, including graphs using only old
        # operations. Exercise both precisions on the default planner and retain
        # the legacy arithmetic check for the primary Float32 path.
        policies = T === Float32 ? (:auto, :legacy) : (:auto,)
        for policy in policies
            outputs = values(fuse(conventional, args...; policy).components)
            host = map(x -> Array(x.data), outputs)
            for i in 1:n
                @test host[1][:, :, i] ≈ expected[i][1] rtol=tolerance atol=tolerance
                @test host[2][:, :, i] ≈ expected[i][2] rtol=tolerance atol=tolerance
                @test host[3][i] ≈ expected[i][3] rtol=tolerance atol=tolerance
                @test host[4][i] ≈ expected[i][4] rtol=tolerance atol=tolerance
                @test host[5][i] ≈ expected[i][5] rtol=tolerance atol=tolerance
                @test host[6][:, :, i] ≈ expected[i][6] rtol=tolerance atol=tolerance
            end
            contaminated = fill(T(NaN), 3, 3, n)
            contaminated[1, 2, :] .= T(Inf)
            identities = Array(
                fuse(identity_matrix, BatchedCuMatrix(CuArray(contaminated)); policy).data
            )
            @test all(identities[:, :, i] == Matrix{T}(I, 3, 3) for i in 1:n)
        end
        @test Array(args[1].data) == matrices
    end
end
