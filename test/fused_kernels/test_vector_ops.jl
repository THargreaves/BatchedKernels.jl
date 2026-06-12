@testitem "Fused matvec + vec-add composition" begin
    # Exercises in one fused kernel:
    #   - matvec on a batched matrix and batched vector
    #   - vec + vec with a shared operand broadcast over the batch
    #   - composition (matvec output feeds vec-add input)
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(A, x, b) = A * x + b
    g(A, x, b) = f.(A, x, b)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        A_cpu = randn(T, D, D, N)
        x_cpu = randn(T, D, N)
        b_cpu = randn(T, D)

        ref = Array{T}(undef, D, N)
        for n in 1:N
            ref[:, n] = A_cpu[:, :, n] * x_cpu[:, n] + b_cpu
        end

        A_gpu = BatchedCuMatrix(CuArray(A_cpu))
        x_gpu = BatchedCuVector(CuArray(x_cpu))
        b_gpu = SharedCuVector(CuArray(b_cpu), N)

        result = @inferred g(A_gpu, x_gpu, b_gpu)
        @test result isa BatchedCuVector{T,D}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end

@testitem "Fused adjoint matvec + vec-sub composition" begin
    # Exercises in one fused kernel:
    #   - adjoint matvec on a shared matrix and batched vector
    #   - vec - vec twice (one batched operand, one shared)
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(y, A, x, c) = y - A' * x - c
    g(y, A, x, c) = f.(y, A, x, c)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        y_cpu = randn(T, D, N)
        x_cpu = randn(T, D, N)
        A_cpu = randn(T, D, D)
        c_cpu = randn(T, D)

        ref = Array{T}(undef, D, N)
        for n in 1:N
            ref[:, n] = y_cpu[:, n] - A_cpu' * x_cpu[:, n] - c_cpu
        end

        y_gpu = BatchedCuVector(CuArray(y_cpu))
        x_gpu = BatchedCuVector(CuArray(x_cpu))
        A_gpu = SharedCuMatrix(CuArray(A_cpu), N)
        c_gpu = SharedCuVector(CuArray(c_cpu), N)

        result = @inferred g(y_gpu, A_gpu, x_gpu, c_gpu)
        @test result isa BatchedCuVector{T,D}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end
