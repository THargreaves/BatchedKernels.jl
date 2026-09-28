@testitem "Backward QR CPU static and residual contracts" tags = [:cpu] begin
    using BatchedKernels, LinearAlgebra, StaticArrays, Random
    include(joinpath(pkgdir(BatchedKernels), "examples", "kalman.jl"))
    Random.seed!(518)
    B = SMatrix{2,3}(randn(Float32, 2, 3))
    r = SVector{2}(randn(2))
    C = SMatrix{4,3}(randn(4, 3))
    q = SVector{4}(randn(4))
    U, s, e = @inferred qr_compress_residual(B, r)
    @test U isa SMatrix{3,3,Float64}
    @test s isa SVector{3,Float64}
    @test e >= 0
    V = @inferred qr_identity_plus(B)
    @test V isa SMatrix{2,2,Float32}
    @test V'V ≈ I + B * B'
    wrapped = UpperTriangular(SMatrix{3,3}(randn(3, 3)))
    W = @inferred qr_identity_plus(wrapped)
    @test W isa SMatrix{3,3,Float64}
    @test W'W ≈ I + wrapped * wrapped'
    U, s, e = @inferred qr_compress_residual(B, r, C, q)
    @test U isa SMatrix{3,3,Float64}
    @test s isa SVector{3,Float64}
    for z in (SVector{3}(randn(3)), zero(SVector{3}))
        @test sum(abs2, B * z - r) + sum(abs2, C * z - q) ≈ sum(abs2, U * z - s) + e
    end
    @test all(diag(U) .>= 0)
    @test istriu(U)
    Z = zero(SMatrix{2,3,Float64})
    U, s, e = qr_compress_residual(Z, r)
    @test U == zero(U)
    @test sum(abs2, s) + e ≈ sum(abs2, r)
    @test_throws DimensionMismatch qr_compress_residual(B, q)
    @test_throws DimensionMismatch qr_compress_residual(B, r, C, r)
    μ = @SVector [0.1, 0.2, 0.3]
    A = one(SMatrix{3,3,Float64})
    UQ = zero(A)
    H = SMatrix{2,3,Float64}(B)
    c = zero(SVector{2,Float64})
    UR = one(SMatrix{2,2,Float64})
    msg = @inferred sqrt_backward_initialise(H, c, UR, r)
    next = @inferred sqrt_backward_step(msg..., A, μ, UQ, H, c, UR, r)
    @test next isa Tuple{SMatrix{3,3,Float64},SVector{3,Float64},Float64}
    @test isfinite(@inferred sqrt_backward_overlap(μ, A, next...))
end

@testitem "Residual QR GPU and heterogeneous result lifetimes" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    const BK = BatchedKernels
    Random.seed!(815)
    for (T, m, n, p, N) in (
        (Float32, 2, 3, 4, 23),
        (Float64, 5, 2, 1, 7),
        (Float32, 16, 20, 20, 3),
        (Float32, 2, 32, 2, 2),
        (Float32, 1, 1, 1, 0),
    )
        host = (randn(T, m, n, N), randn(T, m, N), randn(T, p, n, N), randn(T, p, N))
        if N > 1
            host[1][:, 1, 1] .= 0
            host[3][:, 1, 1] .= 0
            host[1][:, :, 2] .= 0
            host[3][:, :, 2] .= 0
        end
        args = map(
            x -> ndims(x) == 3 ? BatchedCuMatrix(CuArray(x)) : BatchedCuVector(CuArray(x)),
            host,
        )
        out = @inferred fuse(qr_compress_residual, args...; nthreads=64)
        U, s, e = map(x -> Array(x.data), values(out.components))
        @test size(U) == (n, n, N) && size(s) == (n, N) && length(e) == N
        tol = T === Float32 ? 8e-5 : 3e-12
        for k in 1:N
            B, r, C, q = (host[1][:, :, k], host[2][:, k], host[3][:, :, k], host[4][:, k])
            z = randn(T, n)
            @test sum(abs2, B * z - r) + sum(abs2, C * z - q) ≈
                sum(abs2, U[:, :, k] * z - s[:, k]) + e[k] rtol = tol atol = tol
            @test U[:, :, k]'U[:, :, k] ≈ B'B + C'C rtol = tol atol = tol
            @test U[:, :, k]'s[:, k] ≈ B'r + C'q rtol = tol atol = tol
            @test e[k] >= 0 && istriu(U[:, :, k]) && all(diag(U[:, :, k]) .>= 0)
        end
        @test all(Array(a.data) == h for (a, h) in zip(args, host))
        identity_root = Array(fuse(qr_identity_plus, args[1]; nthreads=64).data)
        for k in 1:N
            C = host[1][:, :, k]
            @test identity_root[:, :, k]'identity_root[:, :, k] ≈ I + C * C' rtol = tol atol =
                tol
        end
        if n == 3
            tape = BK.trace(
                qr_compress_residual, BK.InputSpec[BK.input_spec(a) for a in args]
            )
            producer = only(
                i for (i, node) in enumerate(tape.nodes) if
                node isa BK.CallNode && node.fn === qr_compress_residual
            )
            ids = BK.call_result_ids(tape, producer)
            shared = Assignment(tape; nthreads=64)
            plan = BK.plan_memory(tape, shared; D_MAX=4)
            @test plan.slots[ids[2]] != plan.slots[tape.nodes[producer].args[2].id]
            @test plan.slots[ids[2]] != plan.slots[tape.nodes[producer].args[4].id]
            custom = fuse(qr_compress_residual, args...; assignment=shared, nthreads=64)
            @test all(
                Array(x.data) ≈ Array(y.data) for
                (x, y) in zip(values(custom.components), values(out.components))
            )
            @test_throws ArgumentError fuse(
                qr_compress_residual, args...; policy=:legacy, nthreads=64
            )
        end
    end
    # Each unused projection can be omitted independently.
    only_energy(B, r) = last(qr_compress_residual(B, r))
    only_vector(B, r) = qr_compress_residual(B, r)[2]
    only_matrix(B, r) = first(qr_compress_residual(B, r))
    B = randn(Float32, 5, 3, 7)
    r = randn(Float32, 5, 7)
    args = (BatchedCuMatrix(CuArray(B)), BatchedCuVector(CuArray(r)))
    allparts = fuse(qr_compress_residual, args...; nthreads=64)
    for (f, part) in
        zip((only_matrix, only_vector, only_energy), values(allparts.components))
        @test Array(fuse(f, args...; nthreads=64).data) ≈ Array(part.data)
    end
end

@testitem "Normalized backward sequences and particle weights GPU" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    include(joinpath(pkgdir(BatchedKernels), "examples", "kalman.jl"))
    include(joinpath(pkgdir(BatchedKernels), "test", "fusion", "backward_reference.jl"))
    Random.seed!(953)
    for (T, n, m, N, noise) in (
        (Float32, 3, 2, 23, :zero), (Float32, 3, 4, 7, :rankone), (Float64, 5, 2, 7, :full)
    )
        model = map(1:4) do t
            H = randn(T, m, n) / T(3)
            c = randn(T, m) / T(10)
            UR = Matrix(cholesky(Symmetric(Matrix{T}(I, m, m) + H * H')).U)
            y = randn(T, m)
            A = Matrix{T}(I, n, n) + randn(T, n, n) / T(20)
            b = randn(T, n) / T(20)
            UQ = if noise === :zero
                zeros(T, n, n)
            elseif noise === :rankone
                randn(T, 1, n) / T(10)
            else
                Matrix{T}(I, n, n) / T(5)
            end
            (H, c, UR, y, A, b, UQ)
        end
        # Vary observations across particles: normalization must remain per-particle.
        ys = [hcat((x[4] .+ T(k) / T(50) for k in 1:N)...) for x in model]
        function shared(x)
            return if ndims(x) == 2
                SharedCuMatrix(CuArray(x), N)
            else
                SharedCuVector(CuArray(x), N)
            end
        end
        H, c, UR, y, A, b, UQ = model[end]
        msg = @inferred fuse(
            sqrt_backward_initialise,
            shared(H),
            shared(c),
            shared(UR),
            BatchedCuVector(CuArray(ys[end]));
            nthreads=64,
        )
        for t in 3:-1:1
            H, c, UR, y = model[t][1:4]
            A, b, UQ = model[t + 1][5:7]
            msg = @inferred fuse(
                sqrt_backward_step,
                values(msg.components)...,
                shared(A),
                shared(b),
                shared(UQ),
                shared(H),
                shared(c),
                shared(UR),
                BatchedCuVector(CuArray(ys[t]));
                nthreads=64,
            )
        end
        Bs, rs, cs = map(x -> Array(x.data), values(msg.components))
        μ = randn(T, n, N) / T(5)
        U = Matrix{T}(I, n, n) / T(2)
        overlap = @inferred fuse(
            sqrt_backward_overlap,
            BatchedCuVector(CuArray(μ)),
            shared(U),
            values(msg.components)...;
            nthreads=64,
        )
        ovs = Array(overlap.data)
        tol = T === Float32 ? 5e-5 : 3e-12
        for k in 1:N
            models = [
                (x[1], x[2], x[3], ys[t][:, k], x[5], x[6], x[7]) for
                (t, x) in enumerate(model)
            ]
            G, h, V, y = backward_joint_reference(models)
            z = randn(n)
            @test cs[k] - sum(abs2, Bs[:, :, k] * z - rs[:, k]) / 2 ≈
                dense_logpdf(y - h - G * z, V) rtol = tol atol = tol
            @test ovs[k] ≈ dense_logpdf(y - h - G * μ[:, k], V + G * (U'U) * G') rtol = tol atol =
                tol
        end
        # A fixed suffix is shared across candidates in AS and BS.
        B = Bs[:, :, 1]
        r = rs[:, 1]
        logc = cs[1]
        A, b, UQ = model[2][5:7]
        lw = randn(T, N)
        lt = randn(T, N)
        args = (
            BatchedCuVector(CuArray(μ)),
            shared(U),
            shared(A),
            shared(b),
            shared(UQ),
            shared(B),
            shared(r),
            BatchedCuScalar(CuArray(lw)),
            BatchedCuScalar(CuArray(lt)),
        )
        weights = @inferred fuse(sqrt_backward_weight, args...; nthreads=64)
        got = Array(weights.data)
        covargs = (args[1], shared(U'U), args[3], args[4], shared(UQ'UQ), args[6:end]...)
        covweights = fuse(kalman_backward_weight, covargs...; nthreads=64)
        @test Array(covweights.data) ≈ got rtol = tol atol = tol
        for k in 1:N
            @test got[k] ≈
                lw[k] +
                  lt[k] +
                  dense_overlap(A * μ[:, k] + b, A * (U'U) * A' + UQ'UQ, B, r) rtol = tol atol =
                tol
        end
        # Scalar-only calls and broadcast can consume earlier scalar outputs.
        addscalar(a, b) = a + b
        shifted = fuse(
            addscalar, weights, BatchedCuScalar(CuArray(fill(logc, N))); nthreads=64
        )
        @test Array(shifted.data) ≈ got .+ logc
        @test Array(addscalar.(weights, weights).data) ≈ 2got
        @test_throws ArgumentError fuse(
            addscalar,
            weights,
            BatchedCuScalar(CuArray((T === Float32 ? Float64 : Float32).(lw)));
            nthreads=64,
        )
    end
end

@testitem "Backward vector projections and shared calculations" tags = [:gpu] begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    const BK = BatchedKernels
    include(joinpath(pkgdir(BatchedKernels), "examples", "kalman.jl"))
    Random.seed!(665)
    # Delay the residual projection past an unrelated vector producer. Its slot
    # must remain live from the QR producer, not from the later projection.
    delayed(B, r) = begin
        U, s, e = qr_compress_residual(B, r)
        x = r + r
        (U, s, e, x)
    end
    N = 7
    n = 3
    args = (
        BatchedCuMatrix(CuArray(randn(Float32, n, n, N))),
        BatchedCuVector(CuArray(randn(Float32, n, N))),
    )
    t = BK.trace(delayed, BK.InputSpec[BK.input_spec(x) for x in args])
    p = only(
        i for (i, node) in enumerate(t.nodes) if
        node isa BK.CallNode && node.fn === qr_compress_residual
    )
    result = BK.call_result_ids(t, p)[2]
    tmp = only(
        i for (i, node) in enumerate(t.nodes) if node isa BK.CallNode && node.fn === (+)
    )
    order = collect(eachindex(t.nodes))
    deleteat!(order, findfirst(==(result), order))
    insert!(order, findfirst(==(tmp), order) + 1, result)
    assignment = Assignment(t; order, nthreads=64)
    plan = BK.plan_memory(t, assignment; D_MAX=n)
    @test plan.slots[result] != plan.slots[tmp]
    out = fuse(delayed, args...; assignment, nthreads=64)
    normal = fuse(delayed, args...; nthreads=64)
    @test all(
        Array(a.data) ≈ Array(b.data) for
        (a, b) in zip(values(out.components), values(normal.components))
    )
    # Common-only subexpressions are evaluated per particle. Scalar results can
    # pass through a composite input and into a subsequent normalized recursion.
    H = SharedCuMatrix(CuArray(randn(Float32, 2, n)), N)
    c = SharedCuVector(CUDA.zeros(Float32, 2), N)
    UR = SharedCuMatrix(CuArray(Matrix{Float32}(I, 2, 2)), N)
    y = SharedCuVector(CuArray(randn(Float32, 2)), N)
    msg = fuse(sqrt_backward_initialise, H, c, UR, y; nthreads=64)
    A = SharedCuMatrix(CuArray(Matrix{Float32}(I, n, n)), N)
    b = SharedCuVector(CUDA.zeros(Float32, n), N)
    Q = SharedCuMatrix(CUDA.zeros(Float32, n, n), N)
    composite_step(message, A, b, Q, H, c, UR, y) =
        sqrt_backward_step(message..., A, b, Q, H, c, UR, y)
    out = @inferred fuse(composite_step, msg, A, b, Q, H, c, UR, y; nthreads=64)
    split = fuse(sqrt_backward_predict, values(msg.components)..., A, b, Q; nthreads=64)
    split = fuse(
        sqrt_backward_update, values(split.components)..., H, c, UR, y; nthreads=64
    )
    @test all(
        Array(a.data) ≈ Array(b.data) for
        (a, b) in zip(values(out.components), values(split.components))
    )
    # A deterministic forward state is a legitimate singular Gaussian.
    μ = SharedCuVector(CUDA.zeros(Float32, n), N)
    overlap = fuse(sqrt_backward_overlap, μ, Q, values(msg.components)...; nthreads=64)
    Bhost, rhost, chost = map(x -> Array(x.data), values(msg.components))
    @test Array(overlap.data) ≈ chost - vec(sum(abs2, rhost; dims=1)) / 2
    w = BatchedCuScalar(CuArray(fill(-2.0f0, N)))
    weights = fuse(
        kalman_backward_weight,
        μ,
        Q,
        A,
        b,
        Q,
        first(values(msg.components)),
        values(msg.components)[2],
        w,
        w;
        nthreads=64,
    )
    @test Array(weights.data) ≈ -4 .- vec(sum(abs2, rhost; dims=1)) / 2
    identity_scalar(c) = c
    scal = last(values(msg.components))
    @test Array(fuse(identity_scalar, scal; nthreads=64).data) == Array(scal.data)
    @test Array(fuse(identity_scalar, scal; policy=:legacy, nthreads=64).data) ==
        Array(scal.data)
    empty = BatchedCuScalar(CUDA.zeros(Float32, 0))
    @test isempty(fuse(identity_scalar, empty; nthreads=64).data)
end
