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

    # Stage 2: a second fused kernel consumes that Cholesky alongside B.
    h2(C, B) = solve_with.(C, B)
    result = @inferred h2(C_gpu, B_gpu)
    @test result isa BatchedCuMatrix{T,D,D}
    got = Array(result.data)
    @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
end
