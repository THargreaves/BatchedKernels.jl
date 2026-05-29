@testitem "Cholesky Decomposition (in-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("cholesky_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:10
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create symmetric positive definite matrices
            As = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                A_temp = CUDA.rand(Float32, D, D)
                As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
            end
            Us = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_cholesky_inplace!(
                Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode,
            )

            # CPU comparison
            As_cpu = Array(As)
            Us_cpu = similar(As_cpu)
            for i in 1:N
                Us_cpu[:, :, i] = cholesky(As_cpu[:, :, i]).U
            end

            # Replace with upper triangle part
            Us_result = Array(Us)
            for i in 1:N
                Us_result[:, :, i] = UpperTriangular(Us_result[:, :, i])
            end
            max_error = maximum(abs.(Us_result .- Us_cpu))
            @test max_error < 1e-4
        end
    end
end

@testitem "Cholesky Decomposition (out-of-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("cholesky_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create symmetric positive definite matrices
            As = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                A_temp = CUDA.rand(Float32, D, D)
                As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
            end
            Us = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_cholesky_out_of_place!(
                Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode,
            )

            # CPU comparison
            As_cpu = Array(As)
            Us_cpu = similar(As_cpu)
            for i in 1:N
                Us_cpu[:, :, i] = cholesky(As_cpu[:, :, i]).U
            end

            # Replace with upper triangle part
            Us_result = Array(Us)
            for i in 1:N
                Us_result[:, :, i] = UpperTriangular(Us_result[:, :, i])
            end
            max_error = maximum(abs.(Us_result .- Us_cpu))
            @test max_error < 1e-4
        end
    end
end
