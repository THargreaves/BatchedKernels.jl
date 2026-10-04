@testitem "Triangular logabsdet tracing and inference" tags = [:cpu] begin
    using BatchedKernels, LinearAlgebra
    const BK = BatchedKernels

    wrappers = (
        A -> UpperTriangular(A),
        A -> LowerTriangular(A),
        A -> UpperTriangular(A'),
        A -> LowerTriangular(A'),
        A -> UpperTriangular(transpose(A)),
        A -> LowerTriangular(transpose(A)),
    )
    for T in (Float32, Float64), wrap in wrappers
        M = BK.TraceMatrix{T,3,3}
        f(A) = logabsdet(wrap(A))
        @test Core.Compiler.return_type(f, Tuple{M}) ===
            Tuple{BK.TraceScalar{T},BK.TraceScalar{T}}
        tape = BK.trace(f, BK.InputSpec[BK.LeafInput(M, BK.BATCHED)])
        # Ordinary scalar nodes also remain valid for the legacy planner.
        @test !any(n -> n isa BK.ResultNode, tape.nodes)
        @test BK.plan_memory(tape) isa BK.PlannerOutput
        for op in (BK._triangular_logabs, BK._triangular_detsign)
            id = only(
                i for (i, n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === op
            )
            node = tape.nodes[id]
            @test tape.metas[id].type === BK.TraceScalar{T}
            variants = BK.orientation_variants(op, tape.metas[only(node.args).id].type)
            @test only(variants).input_access == (:any,)
            @test only(variants).output_access === :none
        end
    end
end

@testitem "Fused triangular logabsdet semantics" begin
    using BatchedKernels, CUDA, LinearAlgebra

    functions = (
        A -> logabsdet(UpperTriangular(A)),
        A -> logabsdet(LowerTriangular(A)),
        A -> logabsdet(UpperTriangular(A')),
        A -> logabsdet(LowerTriangular(A')),
        A -> logabsdet(UpperTriangular(transpose(A))),
        A -> logabsdet(LowerTriangular(transpose(A))),
    )
    for T in (Float32, Float64)
        diagonals = T[
            2 3 4;
            -2 3 4;
            -2 -3 4;
            0 -3 4;
            -0.0 -3 4;
            Inf -3 4;
            -Inf -3 4;
            NaN 3 4;
            0 Inf 4;
            NaN 0 4;
            floatmax(T) floatmax(T) floatmin(T);
            floatmin(T) floatmin(T) 2;
        ]
        batches = size(diagonals, 1)
        # Every off-diagonal is irrelevant, even if it contains NaN or Inf.
        a = fill(T(NaN), 3, 3, batches)
        a[1, 2, :] .= T(Inf)
        for i in 1:batches, k in 1:3
            a[k, k, i] = diagonals[i, k]
        end
        A = BatchedCuMatrix(CuArray(a))
        selected = T === Float32 ? functions : (last(functions),)
        for f in selected, policy in (:legacy, :auto)
            result = @inferred fuse(f, A; policy)
            magnitude, sign = values(result.components)
            @test magnitude isa BatchedCuScalar{T}
            @test sign isa BatchedCuScalar{T}
            logs, signs = Array(magnitude.data), Array(sign.data)
            tolerance = T === Float32 ? 2.0f-6 : 2e-14
            for i in 1:batches
                expected_log, expected_sign = f(a[:, :, i])
                @test isequal(logs[i], expected_log) ||
                    isapprox(logs[i], expected_log; rtol=tolerance, atol=tolerance)
                @test isequal(signs[i], expected_sign)
            end
            @test isequal(Array(A.data), a)
        end
    end
end

@testitem "Fused generated triangular logabsdet and padded groups" begin
    using BatchedKernels, CUDA, LinearAlgebra, Random

    function generated(A)
        C = cholesky(Symmetric(A))
        l, s = logabsdet(C.L)
        return 2 * l, s, logabsdet(C.U)
    end
    padded(A, B) = (logabsdet(UpperTriangular(A)), B + B)
    rng = Xoshiro(508)
    for (T, n, batches) in ((Float32, 16, 1), (Float64, 3, 5))
        a = Array{T}(undef, n, n, batches)
        for i in 1:batches
            x = randn(rng, T, n, n)
            a[:, :, i] = x * x' + T(n) * I
        end
        A = BatchedCuMatrix(CuArray(a))
        tolerance = T === Float32 ? 3.0f-5 : 2e-12
        # The legacy Cholesky body uses a Float32 shuffle helper. Test Float64
        # generated factors on the hybrid path; both precisions of existing
        # factors are covered under both policies above.
        policies = T === Float32 ? (:legacy, :auto) : (:auto,)
        for policy in policies
            result = @inferred fuse(generated, A; policy, nthreads=64)
            twice_log, sign, upper = values(result.components)
            upper_log, upper_sign = values(upper.components)
            logs, signs = Array(twice_log.data), Array(sign.data)
            ulogs, usigns = Array(upper_log.data), Array(upper_sign.data)
            for i in 1:batches
                ref = generated(a[:, :, i])
                @test logs[i] ≈ ref[1] rtol=tolerance
                @test signs[i] == ref[2]
                @test ulogs[i] ≈ ref[3][1] rtol=tolerance
                @test usigns[i] == ref[3][2]
            end
            @test Array(A.data) == a
        end
    end
    a = reshape(Float32[2, NaN, NaN, Inf, -3, NaN, Inf, Inf, 4], 3, 3, 1)
    A = BatchedCuMatrix(CuArray(a))
    B = BatchedCuMatrix(CUDA.ones(Float32, 5, 5, 1))
    for policy in (:legacy, :auto)
        result = @inferred fuse(padded, A, B; policy)
        determinant, doubled = values(result.components)
        magnitude, sign = values(determinant.components)
        @test only(Array(magnitude.data)) ≈ log(24.0f0)
        @test only(Array(sign.data)) == -1.0f0
        @test Array(doubled.data) == fill(2.0f0, 5, 5, 1)
    end
end
