@testitem "Batch matrix multiplication" begin
    using CUDA
    using Random

    SEED = 1234
    T = Float32
    Ns = [1009, 1024]  # test with prime and power of two batch sizes
    Ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    rng = MersenneTwister(SEED)

    for D in Ds
        for N in Ns
            A = rand(rng, T, D, D, N)
            B = rand(rng, T, D, D, N)
            C = Array{T}(undef, D, D, N)

            A_cuda = cu(A)
            B_cuda = cu(B)
            C_cuda = CuArray{T}(undef, D, D, N)

            batched_matmul!(C_cuda, A_cuda, B_cuda)

            # Compute the expected result on CPU
            for i in 1:N
                C[:, :, i] = A[:, :, i] * B[:, :, i]
            end

            @test Array(C_cuda) ≈ C
        end
    end
end
