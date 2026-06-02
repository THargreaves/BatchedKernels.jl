@testitem "Fused scalar reduction: logdet" begin
    # Matrix → scalar reduction. `logdet(Symmetric(M))` lowers to
    # `logdet(cholesky(M))` and reduces along the warp.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f(M) = logdet(Symmetric(M))
    g(M) = f.(M)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        M_cpu = Array{T}(undef, D, D, N)
        for n in 1:N
            X = randn(T, D, D)
            M_cpu[:, :, n] = X * X' + T(0.1) * I
        end

        ref = Vector{T}(undef, N)
        for n in 1:N
            ref[n] = logdet(Symmetric(M_cpu[:, :, n]))
        end

        M_gpu = BatchedCuMatrix(CuArray(M_cpu))

        result = @inferred g(M_gpu)
        @test result isa BatchedCuScalar{T}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end

@testitem "Fused scalar reduction: sum(abs2, v) and dot(v, v)" begin
    # Vector → scalar reductions. Both lower to the same internal `_norm_sq`
    # primitive, so they should give identical results.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    f_sum(v) = sum(abs2, v)
    f_dot(v) = dot(v, v)
    g_sum(v) = f_sum.(v)
    g_dot(v) = f_dot.(v)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        v_cpu = randn(T, D, N)

        ref = Vector{T}(undef, N)
        for n in 1:N
            ref[n] = sum(abs2, v_cpu[:, n])
        end

        v_gpu = BatchedCuVector(CuArray(v_cpu))

        result_sum = @inferred g_sum(v_gpu)
        @test result_sum isa BatchedCuScalar{T}
        got_sum = Array(result_sum.data)
        @test maximum(abs.(got_sum .- ref)) / maximum(abs.(ref)) < 1e-4

        result_dot = @inferred g_dot(v_gpu)
        @test result_dot isa BatchedCuScalar{T}
        got_dot = Array(result_dot.data)
        @test got_dot ≈ got_sum
    end
end

@testitem "Fused Gaussian log-likelihood" begin
    # End-to-end fused kernel for the multivariate-Gaussian log-density:
    #   ℓ(x; μ, Σ) = -½ (D log 2π + logdet(Σ) + (x - μ)ᵀ Σ⁻¹ (x - μ))
    # Exercises: vec-sub, cholesky on Symmetric, triangular vector solve,
    # logdet, sum(abs2, v), in-kernel scalar arithmetic (`+`, `*`, unary `-`,
    # Number × Scalar), all reduced into a single BatchedCuScalar output.
    using BatchedKernels
    using CUDA
    using LinearAlgebra

    function gauss_ll(x, μ, Σ)
        T = eltype(x)
        D = length(x)
        δ = x - μ
        C = cholesky(Symmetric(Σ))
        return -T(0.5) * (T(D) * log(T(2π)) + logdet(C) + sum(abs2, C.L \ δ))
    end
    g(x, μ, Σ) = gauss_ll.(x, μ, Σ)

    N = 2^9 + 1
    T = Float32

    for D in 2:10
        CUDA.seed!(1234)

        x_cpu = randn(T, D, N)
        μ_cpu = randn(T, D, N)
        Σ_cpu = Array{T}(undef, D, D, N)
        for n in 1:N
            M = randn(T, D, D)
            Σ_cpu[:, :, n] = M * M' + T(0.1) * I
        end

        ref = Vector{T}(undef, N)
        for n in 1:N
            δn = x_cpu[:, n] - μ_cpu[:, n]
            Cn = cholesky(Symmetric(Σ_cpu[:, :, n]))
            ref[n] = -T(0.5) * (T(D) * log(T(2π)) + logdet(Cn) + sum(abs2, Cn.L \ δn))
        end

        x_gpu = BatchedCuVector(CuArray(x_cpu))
        μ_gpu = BatchedCuVector(CuArray(μ_cpu))
        Σ_gpu = BatchedCuMatrix(CuArray(Σ_cpu))

        result = @inferred g(x_gpu, μ_gpu, Σ_gpu)
        @test result isa BatchedCuScalar{T}
        got = Array(result.data)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1e-3
    end
end
