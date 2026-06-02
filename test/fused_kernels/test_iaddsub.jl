@testitem "Wrapper as broadcast output: ±λI ± M with slot-backed parent" begin
    # `I - M`, `M - I`, `2I - M`: the `IAddSubWrapped` tag is returned as the
    # *direct output* of the broadcast. Codegen materialises it through the
    # `_dual_to_single` output transfer, passing `IAddSubGetterMatrix(parent,
    # a, b)` as the source — no extra slot, no extra kernel.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    N = 2^9 + 1
    T = Float32
    D = 4

    CUDA.seed!(1234)
    M_cpu = randn(T, D, D, N)
    M_gpu = BatchedCuMatrix(CuArray(M_cpu))

    @testset "I - M" begin
        f(M) = I - M
        g(M) = f.(M)
        result = @inferred g(M_gpu)
        @test result isa BatchedCuMatrix{T,D,D}
        ref = similar(M_cpu)
        for n in 1:N
            ref[:, :, n] = I - M_cpu[:, :, n]
        end
        @test maximum(abs.(Array(result.data) .- ref)) < T(1e-5)
    end

    @testset "M - I" begin
        f(M) = M - I
        g(M) = f.(M)
        result = @inferred g(M_gpu)
        @test result isa BatchedCuMatrix{T,D,D}
        ref = similar(M_cpu)
        for n in 1:N
            ref[:, :, n] = M_cpu[:, :, n] - I
        end
        @test maximum(abs.(Array(result.data) .- ref)) < T(1e-5)
    end

    @testset "λI - M with λ ≠ 1" begin
        f(M) = 2I - M
        g(M) = f.(M)
        result = @inferred g(M_gpu)
        @test result isa BatchedCuMatrix{T,D,D}
        ref = similar(M_cpu)
        for n in 1:N
            ref[:, :, n] = 2I - M_cpu[:, :, n]
        end
        @test maximum(abs.(Array(result.data) .- ref)) < T(1e-5)
    end
end

@testitem "Wrapper as broadcast output: parent is a CallNode result" begin
    # `I - M*M`: the wrapper's parent isn't a slot-backed input but a matmul
    # result. Codegen must ensure the matmul's slot survives long enough for
    # the wrapped output transfer to read it.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    N = 2^9 + 1
    T = Float32
    D = 4

    CUDA.seed!(1234)
    M_cpu = randn(T, D, D, N)
    M_gpu = BatchedCuMatrix(CuArray(M_cpu))

    f(M) = I - M * M
    g(M) = f.(M)

    result = @inferred g(M_gpu)
    @test result isa BatchedCuMatrix{T,D,D}

    ref = similar(M_cpu)
    for n in 1:N
        ref[:, :, n] = I - M_cpu[:, :, n] * M_cpu[:, :, n]
    end
    @test maximum(abs.(Array(result.data) .- ref)) / maximum(abs.(ref)) < T(1e-3)
end
