@testitem "Fused cholesky + triangular solve (matrix RHS)" begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    # The main consumer of cholesky in fused kernels: factorize, then solve
    # with the U (or L) factor. Both wrappers exercise the same trace path.
    f_U(A, B) = cholesky(A).U \ B
    f_L(A, B) = LowerTriangular(cholesky(A).U') \ B
    g_U(A, B) = f_U.(A, B)
    g_L(A, B) = f_L.(A, B)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        A_cpu = Array{T}(undef, D, D, N)
        for n in 1:N
            X = randn(T, D, D)
            A_cpu[:, :, n] = X * X' + T(0.1) * I
        end
        B_cpu = randn(T, D, D, N)

        ref_U = Array{T}(undef, D, D, N)
        ref_L = Array{T}(undef, D, D, N)
        for n in 1:N
            C = cholesky(A_cpu[:, :, n])
            ref_U[:, :, n] = C.U \ B_cpu[:, :, n]
            ref_L[:, :, n] = C.L \ B_cpu[:, :, n]
        end

        A_gpu = BatchedCuMatrix(CuArray(A_cpu))
        B_gpu = BatchedCuMatrix(CuArray(B_cpu))

        result_U = @inferred g_U(A_gpu, B_gpu)
        @test result_U isa BatchedCuMatrix{T,D,D}
        got_U = Array(result_U.data)
        @test maximum(abs.(got_U .- ref_U)) / maximum(abs.(ref_U)) < 1e-3

        result_L = @inferred g_L(A_gpu, B_gpu)
        @test result_L isa BatchedCuMatrix{T,D,D}
        got_L = Array(result_L.data)
        @test maximum(abs.(got_L .- ref_L)) / maximum(abs.(ref_L)) < 1e-3
    end
end

@testitem "Fused cholesky + triangular solve (vector RHS)" begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(A, b) = cholesky(A).L \ b
    g(A, b) = f.(A, b)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        A_cpu = Array{T}(undef, D, D, N)
        for n in 1:N
            X = randn(T, D, D)
            A_cpu[:, :, n] = X * X' + T(0.1) * I
        end
        b_cpu = randn(T, D, N)

        ref = Array{T}(undef, D, N)
        for n in 1:N
            ref[:, n] = cholesky(A_cpu[:, :, n]).L \ b_cpu[:, n]
        end

        A_gpu = BatchedCuMatrix(CuArray(A_cpu))
        b_gpu = BatchedCuVector(CuArray(b_cpu))

        result = @inferred g(A_gpu, b_gpu)
        @test result isa BatchedCuVector{T,D}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
    end
end

@testitem "Fused cholesky on Symmetric input" begin
    # Verify that wrapping the input in `Symmetric` works (skips stdlib's
    # Hermitian check on the CPU reference and is recognised by the fuser).
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(A, B) = cholesky(Symmetric(A)).U \ B
    g(A, B) = f.(A, B)

    N = 2^9 + 1
    T = Float32
    D = 4

    CUDA.seed!(1234)

    A_cpu = Array{T}(undef, D, D, N)
    for n in 1:N
        X = randn(T, D, D)
        # Construct a slightly-asymmetric SPD-ish input — the upper triangle
        # is what `Symmetric` (default :U uplo) reads.
        A_cpu[:, :, n] = X * X' + T(0.1) * I + T(1e-3) * randn(T, D, D)
    end
    B_cpu = randn(T, D, D, N)

    ref = Array{T}(undef, D, D, N)
    for n in 1:N
        ref[:, :, n] = cholesky(Symmetric(A_cpu[:, :, n])).U \ B_cpu[:, :, n]
    end

    A_gpu = BatchedCuMatrix(CuArray(A_cpu))
    B_gpu = BatchedCuMatrix(CuArray(B_cpu))

    result = @inferred g(A_gpu, B_gpu)
    got = Array(result.data)
    @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
end

@testitem "Composite input: Cholesky from one fused kernel feeds the next" begin
    # Two separate broadcasts. The first produces a batched Cholesky value;
    # the second takes that Cholesky (as a composite input) and uses its `.U`
    # factor inside the kernel. Tests the round-trip of the composite output
    # type — `Cholesky{T, BatchedCuMatrix{T,D,D,…}}` round-trips through the
    # broadcast surface and the second call's `register_wrapped!` field walk.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    factorize(A) = cholesky(A)
    solve_with(C, B) = C.U \ B

    N = 2^9 + 1
    T = Float32
    D = 4

    CUDA.seed!(1234)

    A_cpu = Array{T}(undef, D, D, N)
    for n in 1:N
        X = randn(T, D, D)
        A_cpu[:, :, n] = X * X' + T(0.1) * I
    end
    B_cpu = randn(T, D, D, N)

    ref = Array{T}(undef, D, D, N)
    for n in 1:N
        ref[:, :, n] = cholesky(A_cpu[:, :, n]).U \ B_cpu[:, :, n]
    end

    A_gpu = BatchedCuMatrix(CuArray(A_cpu))
    B_gpu = BatchedCuMatrix(CuArray(B_cpu))

    # Stage 1: cholesky.(A) — fused kernel returns a batched Cholesky value.
    h1(A) = factorize.(A)
    C_gpu = @inferred h1(A_gpu)
    @test eltype(C_gpu) === Cholesky{T,eltype(A_gpu)}

    # Stage 2: a second fused kernel consumes that Cholesky alongside B.
    h2(C, B) = solve_with.(C, B)
    result = @inferred h2(C_gpu, B_gpu)
    @test result isa BatchedCuMatrix{T,D,D}
    got = Array(result.data)
    @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
end

@testitem "Cholesky matrix solve delegates to factor solves" begin
    using BatchedKernels, CUDA, LinearAlgebra, Random
    const BK = BatchedKernels
    direct(A, B) = cholesky(Symmetric(A)) \ B
    factors(A, B) = begin
        C = cholesky(Symmetric(A))
        C.U \ (C.L \ B)
    end
    specs = BK.InputSpec[
        BK.LeafInput(BK.TraceMatrix{Float32,3,3}, BK.BATCHED),
        BK.LeafInput(BK.TraceMatrix{Float32,3,5}, BK.BATCHED),
    ]
    # Identical tapes ensure this convenience dispatch changes no device operations.
    tape = BK.trace(direct, specs)
    @test sprint(show, tape) == sprint(show, BK.trace(factors, specs))
    solves = [n for n in tape.nodes if n isa BK.CallNode && n.fn === (\)]
    @test length(solves) == 2
    @test tape.metas[solves[1].args[1].id].type <: LowerTriangular
    @test tape.metas[solves[2].args[1].id].type <: UpperTriangular
    mismatch = copy(specs)
    mismatch[2] = BK.LeafInput(BK.TraceMatrix{Float32,2,5}, BK.BATCHED)
    @test_throws DimensionMismatch BK.trace(direct, mismatch)
    lower(A, B) = Cholesky(A, 'L', 0) \ B
    @test_throws ArgumentError BK.trace(lower, specs)

    rng = MersenneTwister(681)
    for T in (Float32, Float64)
        N = 19
        A = Array{T}(undef, 3, 3, N)
        for k in 1:N
            X = randn(rng, T, 3, 3)
            A[:, :, k] = X * X' + I
        end
        B = randn(rng, T, 3, 5, N)
        args = (BatchedCuMatrix(CuArray(A)), BatchedCuMatrix(CuArray(B)))
        reference = cat((A[:, :, k] \ B[:, :, k] for k in 1:N)...; dims=3)
        custom = automatic_assignment(
            BK.trace(direct, BK.InputSpec[BK.input_spec(x) for x in args]); nthreads=64
        )
        for options in ((; policy=:auto, nthreads=64), (; assignment=custom))
            result = @inferred fuse(direct, args...; options...)
            @test Array(result.data) ≈ reference rtol = (T === Float32 ? 2e-5 : 1e-12)
        end
        if T === Float32 # Legacy masked Cholesky is Float32-only.
            result = @inferred fuse(direct, args...; policy=:legacy, nthreads=64)
            @test Array(result.data) ≈ reference rtol = 2e-5
        end
        @test Array(args[1].data) == A
        @test Array(args[2].data) == B
    end
end
