@testitem "Fused matmul (square)" begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    mul(A, B) = A * B
    g(A, B) = mul.(A, B)

    # Test parameters
    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        A_cpu = randn(T, D, D, N)
        B_cpu = randn(T, D, D, N)

        ref = Array{T}(undef, D, D, N)
        for n in 1:N
            ref[:, :, n] = A_cpu[:, :, n] * B_cpu[:, :, n]
        end

        A_gpu = BatchedCuMatrix(CuArray(A_cpu))
        B_gpu = BatchedCuMatrix(CuArray(B_cpu))

        result = @inferred g(A_gpu, B_gpu)
        @test result isa BatchedCuMatrix{T,D,D}

        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end

@testitem "Fused matmul (rectangular)" begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    mul(A, B) = A * B
    g(A, B) = mul.(A, B)

    N = 2^9 + 1
    T = Float32

    # (D_M, D_N, D_P) for (M×N)·(N×P) → (M×P). Representative coverage:
    # equal, wider input, taller output, more cols, tall+narrow.
    cases = [
        (3, 3, 3),   # equal
        (3, 4, 2),   # mixed shapes, all different
        (5, 3, 3),   # wider input
        (3, 5, 3),   # common inner dim
        (4, 6, 5),   # padded variant
    ]

    for (D_M, D_N, D_P) in cases
        CUDA.seed!(1234)

        A_cpu = randn(T, D_M, D_N, N)
        B_cpu = randn(T, D_N, D_P, N)

        ref = Array{T}(undef, D_M, D_P, N)
        for n in 1:N
            ref[:, :, n] = A_cpu[:, :, n] * B_cpu[:, :, n]
        end

        A_gpu = BatchedCuMatrix(CuArray(A_cpu))
        B_gpu = BatchedCuMatrix(CuArray(B_cpu))

        result = @inferred g(A_gpu, B_gpu)
        @test result isa BatchedCuMatrix{T,D_M,D_P}
        @test size(result.data) == (D_M, D_P, N)

        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end

@testitem "Fused matmul (mixed batched/shared)" begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    mul(A, B) = A * B
    g(A, B) = mul.(A, B)

    N = 2^9 + 1
    T = Float32

    # Representative dims — both square and rectangular.
    cases = [(3, 3, 3), (3, 4, 2)]

    for (D_M, D_N, D_P) in cases
        CUDA.seed!(1234)

        A_cpu = randn(T, D_M, D_N, N)
        B_cpu = randn(T, D_N, D_P, N)
        A_shared_cpu = randn(T, D_M, D_N)
        B_shared_cpu = randn(T, D_N, D_P)

        # Case 1: A batched, B shared.
        ref1 = Array{T}(undef, D_M, D_P, N)
        for n in 1:N
            ref1[:, :, n] = A_cpu[:, :, n] * B_shared_cpu
        end
        A_gpu = BatchedCuMatrix(CuArray(A_cpu))
        B_gpu_shared = SharedCuMatrix(CuArray(B_shared_cpu), N)

        result1 = @inferred g(A_gpu, B_gpu_shared)
        @test result1 isa BatchedCuMatrix{T,D_M,D_P}
        got1 = Array(result1.data)
        @test maximum(abs.(got1 .- ref1)) / maximum(abs.(ref1)) < 1e-4

        # Case 2: A shared, B batched.
        ref2 = Array{T}(undef, D_M, D_P, N)
        for n in 1:N
            ref2[:, :, n] = A_shared_cpu * B_cpu[:, :, n]
        end
        A_gpu_shared = SharedCuMatrix(CuArray(A_shared_cpu), N)
        B_gpu = BatchedCuMatrix(CuArray(B_cpu))

        result2 = @inferred g(A_gpu_shared, B_gpu)
        @test result2 isa BatchedCuMatrix{T,D_M,D_P}
        got2 = Array(result2.data)
        @test maximum(abs.(got2 .- ref2)) / maximum(abs.(ref2)) < 1e-4
    end
end
