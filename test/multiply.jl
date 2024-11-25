@testitem "Batch matrix multiplication" begin
    using CUDA
    using Random

    SEED = 1234
    T = Float32

    rng = MersenneTwister(SEED)
    A_base = rand(rng, T, 3, 4, 2)
    B_base = rand(rng, T, 4, 2, 2)

    for A_trans in (false, true), B_trans in (false, true)
        A_cpu = A_trans ? permutedims(A_base, (2, 1, 3)) : A_base
        B_cpu = B_trans ? permutedims(B_base, (2, 1, 3)) : B_base

        A = cu(A_cpu)
        B = cu(B_cpu)

        C = batch_matmul(A, B; A_trans, B_trans)
        @test eltype(C) == T

        A_test = A_trans ? transpose(A_cpu[:, :, 1]) : A_cpu[:, :, 1]
        B_test = B_trans ? transpose(B_cpu[:, :, 1]) : B_cpu[:, :, 1]
        @test Array(C[:, :, 1]) ≈ A_test * B_test
    end
end
