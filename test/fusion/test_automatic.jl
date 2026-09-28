@testitem "Automatic orientation planning" tags = [:cpu] begin
    using BatchedKernels, LinearAlgebra
    const BK = BatchedKernels
    specs = BK.InputSpec[BK.LeafInput(BK.TraceMatrix{Float32,3,3}, BK.BATCHED) for _ in 1:2]
    # Symmetrization explicitly needs both orientations; wrappers preserve ownership.
    f(A, B) = symmetric_part(A * B)
    tape = BK.trace(f, specs)
    a = automatic_assignment(tape; nthreads=64)
    p = BK.plan_memory(tape, a; D_MAX=3)
    product = only(
        i for (i, n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === (*)
    )
    sym = only(
        i for
        (i, n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === symmetric_part
    )
    @test a.residences[product] === :dual
    @test a.residences[sym] === :register
    @test p.num_dual_slots == 1
    @test BK.assignment_key(a) == BK.assignment_key(automatic_assignment(tape; nthreads=64))
    @test tape.nodes[product].fn === (*) # planner does not rewrite the source graph
    @test_throws ArgumentError BK.plan_memory(
        tape, automatic_assignment(tape; nthreads=33); D_MAX=3
    )

    transpose_use(A, B) = begin
        C = A * B
        (C * B, B * C')
    end
    t = BK.trace(transpose_use, specs)
    ap = automatic_assignment(t; nthreads=64)
    @test BK.plan_memory(t, ap; D_MAX=3) isa BK.HybridPlannerOutput
    for (i, n) in enumerate(t.nodes)
        if n isa BK.NewNode && t.metas[i].type <: Adjoint
            parent = only(BK.node_refs(n)).id
            @test ap.orientations[i] == BK._flip_assignment(ap.orientations[parent])
        end
    end

    # Explicit mutation retains the existing shared implementation and owner aliases.
    mut(A) = cholesky!(A).U
    mt = BK.trace(mut, specs[1:1])
    ma = automatic_assignment(mt; nthreads=64)
    @test !(:register in values(ma.residences))
    @test BK.plan_memory(mt, ma; D_MAX=3) isa BK.HybridPlannerOutput
end

@testitem "Automatic Joseph forward step and likelihood" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    const BK = BatchedKernels
    include(joinpath(pkgdir(BK), "examples", "kalman.jl"))
    Random.seed!(517)
    # Selected cases cover padded groups, rectangular observations (both directions),
    # partial/full/empty batches, common A/H, particle-specific A/H and all-batched data.
    cases = (
        (Float32, 3, 2, 23, :common),
        (Float64, 3, 5, 13, :all),
        (Float32, 8, 8, 32, :A),
        (Float64, 9, 4, 7, :H),
        (Float32, 16, 6, 9, :all),
        (Float32, 3, 2, 0, :common),
    )
    for (T, d, m, N, lifecycle) in cases
        spd(k) = (X = randn(T, k, k); X * X' / T(k) + Matrix{T}(I, k, k))
        common = (
            randn(T, d),
            spd(d),
            randn(T, d, d) / sqrt(T(d)),
            randn(T, d),
            spd(d),
            randn(T, m, d) / sqrt(T(d)),
            randn(T, m),
            spd(m),
            randn(T, m),
        )
        batchids = if lifecycle === :all
            collect(1:9)
        elseif lifecycle === :A
            [1, 2, 3]
        elseif lifecycle === :H
            [1, 2, 6]
        else
            [1, 2]
        end
        arrays = map(enumerate(common)) do (i, x)
            if i in batchids
                b = ndims(x) == 2 ? repeat(x, 1, 1, N) : repeat(x, 1, N)
                # Vary particles without compromising covariance definiteness.
                for k in 1:N
                    if i in (2, 5, 8)
                        b[:, :, k] .*= T(1 + k / max(N, 1))
                    elseif ndims(x) == 2
                        b[:, :, k] .+= T(0.01k)
                    else
                        b[:, k] .+= T(0.02k)
                    end
                end
                b
            else
                copy(x)
            end
        end
        args = Tuple(
            if i in batchids
                if ndims(common[i]) == 2
                    BatchedCuMatrix(CuArray(x))
                else
                    BatchedCuVector(CuArray(x))
                end
            else
                if ndims(x) == 2
                    SharedCuMatrix(CuArray(x), N)
                else
                    SharedCuVector(CuArray(x), N)
                end
            end for (i, x) in enumerate(arrays)
        )
        out = @inferred fuse(joseph_kalman_step, args...; nthreads=64)
        got = map(x -> Array(x.data), values(getfield(out, :components)))
        @test size(got[1]) == (d, N)
        @test size(got[2]) == (d, d, N)
        @test size(got[3]) == (N,)
        refs = map(1:N) do k
            scalar = map(enumerate(arrays)) do (i, x)
                i in batchids ? (ndims(common[i]) == 2 ? x[:, :, k] : x[:, k]) : x
            end
            # Independent reference uses the original covariance/gain equations.
            μ, P, A, b, Q, H, c, R, y = scalar
            μp = A * μ + b
            Pp = A * P * A' + Q
            e = y - H * μp - c
            S = Symmetric(H * Pp * H' + R)
            K = (Pp * H') / S
            J = I - K * H
            Pf = J * Pp * J' + K * R * K'
            ll = -T(0.5) * (T(m) * log(T(2) * T(π)) + logdet(S) + dot(e, S \ e))
            (μp + K * e, (Pf + Pf') / 2, ll)
        end
        tol = T === Float32 ? 5e-5 : 2e-12
        for k in 1:N
            @test got[1][:, k] ≈ refs[k][1] rtol = tol atol = tol
            @test got[2][:, :, k] ≈ refs[k][2] rtol = tol atol = tol
            @test got[3][k] ≈ refs[k][3] rtol = tol atol = tol
            @test issymmetric(got[2][:, :, k])
            @test minimum(eigvals(Symmetric(got[2][:, :, k]))) >= -tol
        end
        @test all(Array(arg.data) == x for (arg, x) in zip(args, arrays))
        entry = BK._ensure_compiled!(joseph_kalman_step, args; nthreads=64)
        @test entry === BK._ensure_compiled!(joseph_kalman_step, args; nthreads=64)
        if T === Float32 && d == 3 && N > 0
            legacy = @inferred fuse(
                joseph_kalman_step, args...; policy=:legacy, nthreads=64
            )
            legacy_arrays = map(x -> Array(x.data), values(getfield(legacy, :components)))
            @test all(
                isapprox(a, b; rtol=tol, atol=tol) for (a, b) in zip(got, legacy_arrays)
            )
            @test entry !== BK._ensure_compiled!(
                joseph_kalman_step, args; policy=:legacy, nthreads=64
            )
            @test_throws ArgumentError fuse(joseph_kalman_step, args...; policy=:unknown)
        end
    end
end

@testitem "Masked vector output and full-warp reductions" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra
    # Short output vectors use D_MAX-strided shared storage but packed global storage.
    f(A, x) = (A * x, sum(abs2, x))
    for (m, n, N) in ((2, 3, 23), (3, 32, 3))
        A = reshape(sin.(Float32.(1:(m * n * N))), m, n, N)
        x = reshape(cos.(Float32.(1:(n * N))), n, N)
        out = @inferred fuse(
            f, BatchedCuMatrix(CuArray(A)), BatchedCuVector(CuArray(x)); nthreads=64
        )
        ys, ss = map(v -> Array(v.data), values(getfield(out, :components)))
        for k in 1:N
            @test ys[:, k] ≈ A[:, :, k] * x[:, k] atol = 2e-5
            @test ss[k] ≈ sum(abs2, x[:, k])
        end
    end
end

@testitem "Joseph recursion with singular state covariance" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra
    const BK = BatchedKernels
    include(joinpath(pkgdir(BK), "examples", "kalman.jl"))
    for T in (Float32, Float64)
        N = 3
        d = 3
        m = 2
        μ0 = T[1, -2, 3]
        P0 = Matrix(Diagonal(T[1e4, 1e-2, 0]))
        A = Matrix{T}(I, d, d)
        b = zeros(T, d)
        Q = zeros(T, d, d)
        H = T[1 0 0; 0 1 0]
        c = zeros(T, m)
        R = Matrix(Diagonal(T[1e-2, 1e-6]))
        y = T[2, -1]
        μ = BatchedCuVector(CuArray(repeat(μ0, 1, N)))
        P = BatchedCuMatrix(CuArray(repeat(P0, 1, 1, N)))
        common = (
            SharedCuMatrix(CuArray(A), N),
            SharedCuVector(CuArray(b), N),
            SharedCuMatrix(CuArray(Q), N),
            SharedCuMatrix(CuArray(H), N),
            SharedCuVector(CuArray(c), N),
            SharedCuMatrix(CuArray(R), N),
            SharedCuVector(CuArray(y), N),
        )
        refμ, refP = Float64.(μ0), Float64.(P0)
        cached = nothing
        for step in 1:8
            # Common observations change without retracing or caching their contents.
            y .+= T(0.01)
            copyto!(common[end].data, y)
            args = (μ, P, common...)
            entry = BK._ensure_compiled!(joseph_kalman_step, args; nthreads=64)
            @test cached === nothing || entry === cached
            cached = entry
            out = fuse(joseph_kalman_step, args...; nthreads=64)
            μ, P, ll = values(getfield(out, :components))
            refμ, refP, refll = joseph_kalman_step(
                refμ,
                refP,
                Float64.(A),
                Float64.(b),
                Float64.(Q),
                Float64.(H),
                Float64.(c),
                Float64.(R),
                Float64.(y),
            )
            tol = T === Float32 ? 3e-4 : 1e-10
            @test Array(μ.data)[:, 1] ≈ refμ rtol = tol atol = tol
            @test Array(P.data)[:, :, 1] ≈ refP rtol = tol atol = tol * 1e-4
            @test Array(ll.data)[1] ≈ refll rtol = tol atol = tol
            @test all(isfinite, Array(ll.data))
        end
    end
    # Batch consistency must be checked even when the shared input comes first.
    f(A, x) = A * x
    @test_throws ErrorException fuse(
        f, SharedCuMatrix(CUDA.ones(3, 3), 4), BatchedCuVector(CUDA.ones(3, 2))
    )
end

@testitem "Unsupported element types are rejected" tags = [:gpu] begin
    using BatchedKernels, CUDA
    product(A, B) = A * B
    A = BatchedCuMatrix(CUDA.rand(Float16, 3, 3, 4))
    for policy in (:auto, :legacy)
        @test_throws ArgumentError fuse(product, A, A; policy)
    end
end
