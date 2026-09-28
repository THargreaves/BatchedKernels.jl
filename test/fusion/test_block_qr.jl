@testitem "Block QR CPU static contract" tags = [:cpu] begin
    using BatchedKernels, LinearAlgebra, StaticArrays, Random
    include(joinpath(pkgdir(BatchedKernels), "examples", "kalman.jl"))
    Random.seed!(608)
    A = SMatrix{2,2}(randn(Float32, 2, 2))
    B = SMatrix{3,2}(randn(3, 2))
    C = SMatrix{3,3}(randn(3, 3))
    parts = @inferred qr_upper_blocks(A, B, C)
    @test parts isa Tuple{SMatrix{2,2,Float64},SMatrix{2,3,Float64},SMatrix{3,3,Float64}}
    R = [parts[1] parts[2]; zeros(3, 2) parts[3]]
    M = [A zeros(2, 3); B C]
    @test R'R ≈ M'M
    @test all(diag(R) .>= 0)
    root = @inferred qr_upper_stack(B, A)
    @test root isa SMatrix{2,2,Float64}
    @test root'root ≈ A'A + B'B
    # Respect triangular wrappers even when their parent has a dirty lower half.
    wrapped = @inferred qr_upper_stack(B, UpperTriangular(A))
    @test wrapped isa SMatrix{2,2}
    @test wrapped'wrapped ≈ UpperTriangular(A)'UpperTriangular(A) + B'B
    Z = zero(SMatrix{2,2,Float64})
    rankone = @SMatrix [0.0 3.0; 0.0 0.0]
    r = @inferred qr_upper_stack(Z, rankone)
    @test r'r ≈ rankone'rankone
    @test r[1, 2] == 3
    @test qr_upper_stack(Z, Z) == Z
    @test qr_upper_stack(zero(SMatrix{0,2,Float64}), rankone)' *
          qr_upper_stack(zero(SMatrix{0,2,Float64}), rankone) ≈ rankone'rankone
    @test_throws DimensionMismatch qr_upper_stack(B, C)
    @test_throws DimensionMismatch qr_upper_blocks(A, B, A)
    μ = SVector(0.1, 0.2, 0.3)
    U = one(SMatrix{3,3,Float64})
    H = SMatrix{2,3}(B')
    args = (
        μ,
        U,
        C,
        zero(μ),
        SMatrix{1,3}(1.0, 0.0, 0.0),
        H,
        SVector(0.0, 0.0),
        one(SMatrix{2,2,Float64}),
        SVector(1.0, 2.0),
    )
    μf, Uf, ll = @inferred srkf_step(args...)
    @test μf isa SVector{3,Float64}
    @test Uf isa SMatrix{3,3,Float64}
    @test ll isa Float64
    ref = joseph_kalman_step(
        args[1],
        U'U,
        args[3],
        args[4],
        args[5]'args[5],
        H,
        args[7],
        args[8]'args[8],
        args[9],
    )
    @test μf ≈ ref[1]
    @test Uf'Uf ≈ ref[2]
    @test ll ≈ ref[3]
end

@testitem "Multi-result QR storage lifetimes" tags = [:cpu] begin
    using BatchedKernels, LinearAlgebra
    const BK = BatchedKernels
    specs = BK.InputSpec[
        BK.LeafInput(BK.TraceMatrix{Float32,m,n}, BK.BATCHED) for
        (m, n) in ((2, 2), (3, 2), (3, 3))
    ]
    t = BK.trace(qr_upper_blocks, specs)
    id = only(
        i for (i, n) in enumerate(t.nodes) if n isa BK.CallNode && n.fn === qr_upper_blocks
    )
    results = BK.call_result_ids(t, id)
    @test length(results) == 3
    @test all(t.nodes[r] isa BK.ResultNode for r in results)
    @test all(BK.canonical_storage_owners(t)[r] == r for r in results)
    auto = automatic_assignment(t; nthreads=64)
    @test all(auto.residences[r] === :register for r in results)
    @test BK.plan_memory(t, auto; D_MAX=3).num_dual_slots == 0
    shared = Assignment(t; nthreads=64)
    plan = BK.plan_memory(t, shared; D_MAX=3)
    @test length(unique(plan.slots[r] for r in results)) == 3
    @test all(plan.slots[r] != plan.slots[a.id] for r in results for a in t.nodes[id].args)
    @test_throws ArgumentError BK.schedule(t)
    @test_throws ArgumentError BK.plan_memory(t)
    bad = copy(shared.order)
    bad[id], bad[results[1]] = bad[results[1]], bad[id]
    @test_throws ArgumentError BK.plan_memory(t, Assignment(t; order=bad); D_MAX=3)
    @test_throws ArgumentError BK.plan_memory(
        t, Assignment(t; variants=Dict(id => :legacy)); D_MAX=3
    )
    @test_throws ArgumentError BK.plan_memory(
        t,
        Assignment(
            t;
            residences=Dict(results[1] => :register),
            orientations=Dict(results[1] => :row),
            variants=Dict(id => :qr_blocks_col),
        );
        D_MAX=3,
    )
    # A delayed projection does not delay allocation of the corresponding result.
    f(A, B, C) = begin
        U, V, W = qr_upper_blocks(A, B, C)
        X = C + C
        (U, V, W, X)
    end
    t2 = BK.trace(f, specs)
    producer = only(
        i for (i, n) in enumerate(t2.nodes) if n isa BK.CallNode && n.fn === qr_upper_blocks
    )
    resultids = BK.call_result_ids(t2, producer)
    order = collect(eachindex(t2.nodes))
    delayed = popat!(order, findfirst(==(resultids[3]), order))
    tmp = only(i for (i, n) in enumerate(t2.nodes) if n isa BK.CallNode && n.fn === (+))
    insert!(order, findfirst(==(tmp), order) + 1, delayed)
    a2 = Assignment(t2; order, nthreads=64)
    p2 = BK.plan_memory(t2, a2; D_MAX=3)
    pos = Dict(i => p for (p, i) in enumerate(order))
    @test BK._result_birth(t2, delayed, pos) == pos[producer]
    @test p2.slots[delayed] != p2.slots[tmp]
    @test all(p2.slots[delayed] != p2.slots[a.id] for a in t2.nodes[producer].args)
end

@testitem "Block QR GPU factors and storage policies" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    const BK = BatchedKernels
    Random.seed!(921)
    cases = (
        (Float32, 2, 3, 23), (Float64, 5, 2, 7), (Float32, 16, 20, 3), (Float64, 2, 32, 2)
    )
    for (T, m, n, N) in cases
        host = (randn(T, m, m, N), randn(T, n, m, N), randn(T, n, n, N))
        args = map(x -> BatchedCuMatrix(CuArray(x)), host)
        out = @inferred fuse(
            qr_upper_blocks,
            args...;
            nthreads=64,
            shared_memory=max(m, n) == 32 ? :dynamic : :static,
        )
        got = map(x -> Array(x.data), values(out.components))
        tol = T === Float32 ? 4e-5 : 2e-12
        for k in 1:N
            A, B, C = map(x -> x[:, :, k], host)
            R11, R12, R22 = map(x -> x[:, :, k], got)
            R = [R11 R12; zeros(T, n, m) R22]
            M = [A zeros(T, m, n); B C]
            @test R'R ≈ M'M rtol = tol atol = tol
            @test istriu(R)
            @test all(diag(R) .>= 0)
            ref = qr_upper_blocks(A, B, C)
            @test all(
                isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip((R11, R12, R22), ref)
            )
        end
        @test all(Array(x.data) == y for (x, y) in zip(args, host))
        # Both access contracts remain supported, including full-group masks,
        # non-power-of-two extents and logical stacks larger than a warp.
        tape = BK.trace(qr_upper_blocks, BK.InputSpec[BK.input_spec(x) for x in args])
        producer = only(
            i for (i, node) in enumerate(tape.nodes) if
            node isa BK.CallNode && node.fn === qr_upper_blocks
        )
        for variant in (:qr_blocks_row, :qr_blocks_col)
            # The all-dual D32 fixture needs one warp/block to fit the arena.
            threads = max(m, n) == 32 ? 32 : 64
            assignment = Assignment(
                tape; variants=Dict(producer => variant), nthreads=threads
            )
            forced = fuse(qr_upper_blocks, args...; assignment, shared_memory=:dynamic)
            fg = map(x -> Array(x.data), values(forced.components))
            @test all(isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip(got, fg))
        end
        if m == 2 && n == 3
            t = BK.trace(qr_upper_blocks, BK.InputSpec[BK.input_spec(x) for x in args])
            a = Assignment(t; nthreads=64)
            forced = fuse(qr_upper_blocks, args...; assignment=a, shared_memory=:dynamic)
            fg = map(x -> Array(x.data), values(forced.components))
            @test all(isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip(got, fg))
            auto = automatic_assignment(t; nthreads=64)
            residences = copy(auto.residences)
            for (i, node) in enumerate(t.nodes)
                node isa BK.ResultNode && (residences[i] = :single)
            end
            singles = Assignment(
                t;
                residences,
                orientations=auto.orientations,
                variants=auto.variants,
                nthreads=64,
            )
            sg = map(
                x -> Array(x.data),
                values(fuse(qr_upper_blocks, args...; assignment=singles).components),
            )
            @test all(isapprox(x, y; rtol=tol, atol=tol) for (x, y) in zip(got, sg))
            @test_throws ArgumentError fuse(
                qr_upper_blocks, args...; policy=:legacy, nthreads=64
            )
            onlyposterior(A, B, C) = last(qr_upper_blocks(A, B, C))
            @test Array(fuse(onlyposterior, args...; nthreads=64).data) ≈ got[3]
        end
    end
    # Scaled norms, zero pivots, a zero diagonal with a nonzero trailing row,
    # and rank-deficient inputs. Compare Gram matrices after rescaling.
    for T in (Float32, Float64)
        scales = T === Float32 ? T[1e-40, 1e-25, 1, 1e25] : T[1e-310, 1e-200, 1, 1e200]
        for scale in scales
            A = scale * T[0 3 1; 0 0 0; 0 0 2]
            B = zeros(T, 2, 3)
            args = (
                BatchedCuMatrix(CuArray(repeat(B, 1, 1, 5))),
                BatchedCuMatrix(CuArray(repeat(A, 1, 1, 5))),
            )
            got = Array(fuse(qr_upper_stack, args...; nthreads=64).data)[:, :, 1] / scale
            @test all(isfinite, got)
            @test got'got ≈ (A / scale)' * (A / scale) rtol = 1e-5
            @test got[1, 2] ≈ T(3)
            @test istriu(got)
            legacy = Array(fuse(qr_upper_stack, args...; policy=:legacy, nthreads=64).data)
            @test legacy[:, :, 1] / scale ≈ got rtol = 1e-5
        end
        Z = BatchedCuMatrix(CUDA.zeros(T, 3, 3, 3))
        @test all(iszero, Array(fuse(qr_upper_stack, Z, Z; nthreads=64).data))
    end
end

@testitem "Automatic SRKF step and likelihood" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    const BK = BatchedKernels
    include(joinpath(pkgdir(BK), "examples", "kalman.jl"))
    Random.seed!(827)
    cases = (
        (Float32, 3, 2, 1, 23, :common),
        (Float64, 3, 5, 2, 7, :all),
        (Float32, 8, 4, 8, 9, :A),
        (Float64, 9, 4, 3, 3, :H),
        (Float32, 20, 16, 2, 3, :common),
        (Float32, 3, 2, 1, 0, :common),
    )
    for (T, n, m, r, N, lifecycle) in cases
        U = Matrix(UpperTriangular(randn(T, n, n)))
        U[n, n] = 0 # rank-deficient state root is legal
        UR = Matrix(UpperTriangular(randn(T, m, m))) + T(m) * Matrix{T}(I, m, m)
        UR = qr_upper_stack(zeros(T, m, m), UR)
        common = (
            randn(T, n),
            U,
            randn(T, n, n) / sqrt(T(n)),
            randn(T, n),
            randn(T, r, n) / sqrt(T(n)),
            randn(T, m, n) / sqrt(T(n)),
            randn(T, m),
            UR,
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
        host = map(enumerate(common)) do (i, x)
            if i in batchids
                (ndims(x) == 1 ? repeat(x, 1, N) : repeat(x, 1, 1, N))
            else
                copy(x)
            end
        end
        args = Tuple(
            if i in batchids
                if ndims(common[i]) == 1
                    BatchedCuVector(CuArray(x))
                else
                    BatchedCuMatrix(CuArray(x))
                end
            else
                if ndims(common[i]) == 1
                    SharedCuVector(CuArray(x), N)
                else
                    SharedCuMatrix(CuArray(x), N)
                end
            end for (i, x) in enumerate(host)
        )
        result = @inferred fuse(srkf_step, args...; nthreads=64)
        μs, Us, lls = map(x -> Array(x.data), values(result.components))
        @test size(μs) == (n, N) && size(Us) == (n, n, N) && size(lls) == (N,)
        tol = T === Float32 ? 2e-4 : 2e-11
        for k in 1:N
            xs = map(enumerate(host)) do (i, x)
                i in batchids ? (ndims(common[i]) == 1 ? x[:, k] : x[:, :, k]) : x
            end
            μ, U, A, b, UQ, H, c, UR, y = xs
            # Covariance-form Joseph update is independent of block QR.
            ref = joseph_kalman_step(μ, U'U, A, b, UQ'UQ, H, c, UR'UR, y)
            @test μs[:, k] ≈ ref[1] rtol = tol atol = tol
            @test Us[:, :, k]'Us[:, :, k] ≈ ref[2] rtol = tol atol = tol
            @test lls[k] ≈ ref[3] rtol = tol atol = tol
            @test istriu(Us[:, :, k]) && all(diag(Us[:, :, k]) .>= 0)
        end
        @test all(Array(x.data) == y for (x, y) in zip(args, host))
        entry = BK._ensure_compiled!(srkf_step, args; nthreads=64)
        @test entry === BK._ensure_compiled!(srkf_step, args; nthreads=64)
    end
end

@testitem "SRKF recursion with zero process noise" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra
    const BK = BatchedKernels
    include(joinpath(pkgdir(BK), "examples", "kalman.jl"))
    for T in (Float32, Float64)
        N = 3
        μ0 = T[1, -2, 3]
        U0 = Matrix(Diagonal(T[100, 0.1, 0]))
        A = Matrix{T}(I, 3, 3)
        b = zeros(T, 3)
        UQ = zeros(T, 3, 3)
        H = T[1 0 0; 0 1 0]
        c = zeros(T, 2)
        UR = Matrix(Diagonal(T[1e-2, 1e-3]))
        y = T[2, -1]
        common = (
            SharedCuMatrix(CuArray(A), N),
            SharedCuVector(CuArray(b), N),
            SharedCuMatrix(CuArray(UQ), N),
            SharedCuMatrix(CuArray(H), N),
            SharedCuVector(CuArray(c), N),
            SharedCuMatrix(CuArray(UR), N),
            SharedCuVector(CuArray(y), N),
        )
        μ = BatchedCuVector(CuArray(repeat(μ0, 1, N)))
        U = BatchedCuMatrix(CuArray(repeat(U0, 1, 1, N)))
        refμ = Float64.(μ0)
        refP = Float64.(U0)'Float64.(U0)
        cached = nothing
        for step in 1:6
            y .+= T(0.001)
            copyto!(common[end].data, y)
            args = (μ, U, common...)
            entry = BK._ensure_compiled!(srkf_step, args; nthreads=64)
            @test cached === nothing || entry === cached
            cached = entry
            out = fuse(srkf_step, args...; nthreads=64)
            μ, U, ll = values(out.components)
            refμ, refP, refll = joseph_kalman_step(
                refμ,
                refP,
                Float64.(A),
                Float64.(b),
                Float64.(UQ),
                Float64.(H),
                Float64.(c),
                Float64.(UR)'Float64.(UR),
                Float64.(y),
            )
            root = Array(U.data)[:, :, 1]
            tol = T === Float32 ? 3e-4 : 1e-10
            @test Array(μ.data)[:, 1] ≈ refμ rtol = tol atol = tol
            # The 1e4 root-scale ratio gives ~4.3e-4 relative covariance
            # error even with CPU Float32 QR (GPU ~3.3e-4). Keep the Float64
            # reference, with a conditioning-appropriate bound for this case.
            covariance_tol = T === Float32 ? 1e-3 : tol
            @test root'root ≈ refP rtol = covariance_tol atol = tol * 1e-6
            @test Array(ll.data)[1] ≈ refll rtol = tol atol = tol
            @test all(isfinite, Array(ll.data))
        end
    end
end

@testitem "QR scaled reflector numerical edges" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra
    const BK = BatchedKernels
    # The Float64 maximum uses unsigned IEEE ordering to avoid FP64 comparisons.
    for T in (Float32, Float64)
        values = T[0, -0.0, nextfloat(zero(T)), floatmin(T), -1, 3, floatmax(T), Inf, NaN]
        @test all(
            isequal(BK._qr_absmax(abs(a), b), max(abs(a), abs(b))) for a in values,
            b in values
        )
        scales = T === Float32 ? T[1e-40, 1e-25, 1, 1e25] : T[1e-310, 1e-200, 1, 1e200]
        for scale in scales
            A = scale * T[1 -2 3; 2 1 -1]
            B = scale * T[2 1 -1; 0 3 2; 0 0 -2]
            args = map(x -> BatchedCuMatrix(CuArray(repeat(x, 1, 1, 7))), (A, B))
            tape = BK.trace(qr_upper_stack, BK.InputSpec[BK.input_spec(x) for x in args])
            producer = only(
                i for (i, node) in enumerate(tape.nodes) if
                node isa BK.CallNode && node.fn === qr_upper_stack
            )
            reference =
                (Float64.(A) / Float64(scale))' * (Float64.(A) / Float64(scale)) +
                (Float64.(B) / Float64(scale))' * (Float64.(B) / Float64(scale))
            tol = T === Float32 ? (scale < floatmin(T) ? 2e-4 : 2e-5) : 2e-12
            for variant in (:qr_stack_row, :qr_stack_col)
                assignment = Assignment(
                    tape; variants=Dict(producer => variant), nthreads=64
                )
                roots = Array(fuse(qr_upper_stack, args...; assignment).data)
                @test all(isfinite, roots)
                for k in axes(roots, 3)
                    root = Float64.(roots[:, :, k]) / Float64(scale)
                    @test root'root ≈ reference rtol = tol atol = tol
                    @test istriu(root) && all(diag(root) .>= 0)
                end
            end
        end
        # Mask the dirty lower triangle before loading a column-owned fragment.
        A = T[1 2 3; 4 5 6]
        B = T[2 1 -1; 99 3 2; 88 77 -2]
        args = map(x -> BatchedCuMatrix(CuArray(repeat(x, 1, 1, 7))), (A, B))
        wrapped(A, B) = qr_upper_stack(A, UpperTriangular(B))
        roots = Array(fuse(wrapped, args...; nthreads=64).data)
        reference = A'A + UpperTriangular(B)'UpperTriangular(B)
        @test roots[:, :, 1]'roots[:, :, 1] ≈ reference
    end
end
