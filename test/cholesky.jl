@testitem "Batch Cholesky" begin
    using CUDA
    using Random
    using LinearAlgebra

    SEED = 1234
    T = Float32
    Ns = [1009, 1024]  # test with prime and power of two batch sizes
    Ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    rng = MersenneTwister(SEED)

    for D in Ds
        for N in Ns
            A = rand(rng, T, D, D, N)
            U = Array{T}(undef, D, D, N)

            # Make PSD
            for i in 1:N
                A[:, :, i] = A[:, :, i] * A[:, :, i]' + I
            end

            A_cuda = cu(A)
            U_cuda = CuArray{T}(undef, D, D, N)

            batched_cholesky!(U_cuda, A_cuda)

            # Compute the expected result on CPU
            for i in 1:N
                U[:, :, i] = cholesky(A[:, :, i]).U
            end

            U_cpu = Array(U_cuda)
            # Set lower triangle to zero
            @inbounds for n in 1:N
                for i in 1:D
                    for j in 1:(i - 1)
                        U_cpu[i, j, n] = 0.0f0
                    end
                end
            end

            @test U_cpu ≈ U
        end
    end
end
