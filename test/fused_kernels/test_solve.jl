@testitem "Fused triangular solve: UpperTriangular \\ matrix" begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(U, B) = UpperTriangular(U) \ B
    g(U, B) = f.(U, B)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        # Pad the diagonal so the triangular factor is well-conditioned.
        U_cpu = randn(T, D, D, N)
        for n in 1:N
            U_cpu[:, :, n] += T(D) * I
        end
        B_cpu = randn(T, D, D, N)

        ref = Array{T}(undef, D, D, N)
        for n in 1:N
            ref[:, :, n] = UpperTriangular(U_cpu[:, :, n]) \ B_cpu[:, :, n]
        end

        U_gpu = BatchedCuMatrix(CuArray(U_cpu))
        B_gpu = BatchedCuMatrix(CuArray(B_cpu))

        result = @inferred g(U_gpu, B_gpu)
        @test result isa BatchedCuMatrix{T,D,D}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
    end
end

@testitem "Fused triangular solve: LowerTriangular \\ matrix" begin
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(L, B) = LowerTriangular(L) \ B
    g(L, B) = f.(L, B)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        L_cpu = randn(T, D, D, N)
        for n in 1:N
            L_cpu[:, :, n] += T(D) * I
        end
        B_cpu = randn(T, D, D, N)

        ref = Array{T}(undef, D, D, N)
        for n in 1:N
            ref[:, :, n] = LowerTriangular(L_cpu[:, :, n]) \ B_cpu[:, :, n]
        end

        L_gpu = BatchedCuMatrix(CuArray(L_cpu))
        B_gpu = BatchedCuMatrix(CuArray(B_cpu))

        result = @inferred g(L_gpu, B_gpu)
        @test result isa BatchedCuMatrix{T,D,D}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
    end
end

@testitem "Fused triangular solve: LowerTriangular \\ vector" begin
    # Vector RHS is the path the Kalman likelihood takes (`C.L \ δ`).
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(L, b) = LowerTriangular(L) \ b
    g(L, b) = f.(L, b)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        L_cpu = randn(T, D, D, N)
        for n in 1:N
            L_cpu[:, :, n] += T(D) * I
        end
        b_cpu = randn(T, D, N)

        ref = Array{T}(undef, D, N)
        for n in 1:N
            ref[:, n] = LowerTriangular(L_cpu[:, :, n]) \ b_cpu[:, n]
        end

        L_gpu = BatchedCuMatrix(CuArray(L_cpu))
        b_gpu = BatchedCuVector(CuArray(b_cpu))

        result = @inferred g(L_gpu, b_gpu)
        @test result isa BatchedCuVector{T,D}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
    end
end

@testitem "Fused triangular solve: shared LHS, batched RHS" begin
    # A common pattern: factorize once on the host, reuse across the batch.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f_U(U, B) = UpperTriangular(U) \ B
    f_L(L, B) = LowerTriangular(L) \ B
    g_U(U, B) = f_U.(U, B)
    g_L(L, B) = f_L.(L, B)

    N = 2^9 + 1
    T = Float32
    D = 4

    CUDA.seed!(1234)

    U_cpu = randn(T, D, D) + T(D) * I
    L_cpu = randn(T, D, D) + T(D) * I
    B_cpu = randn(T, D, D, N)

    ref_U = Array{T}(undef, D, D, N)
    ref_L = Array{T}(undef, D, D, N)
    for n in 1:N
        ref_U[:, :, n] = UpperTriangular(U_cpu) \ B_cpu[:, :, n]
        ref_L[:, :, n] = LowerTriangular(L_cpu) \ B_cpu[:, :, n]
    end

    U_gpu = SharedCuMatrix(CuArray(U_cpu), N)
    L_gpu = SharedCuMatrix(CuArray(L_cpu), N)
    B_gpu = BatchedCuMatrix(CuArray(B_cpu))

    result_U = @inferred g_U(U_gpu, B_gpu)
    @test result_U isa BatchedCuMatrix{T,D,D}
    got_U = Array(result_U.data)
    @test maximum(abs.(got_U .- ref_U)) / maximum(abs.(ref_U)) < 1e-3

    result_L = @inferred g_L(L_gpu, B_gpu)
    @test result_L isa BatchedCuMatrix{T,D,D}
    got_L = Array(result_L.data)
    @test maximum(abs.(got_L .- ref_L)) / maximum(abs.(ref_L)) < 1e-3
end
