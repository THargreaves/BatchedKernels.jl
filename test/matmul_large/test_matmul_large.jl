@testitem "Matrix Multiplication (large)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("matmul_large_kernels.jl")

    # Test parameters
    N = 2^12

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    # Test all four combinations
    test_cases = [(false, false), (true, false), (false, true), (true, true)]

    # Accuracy tests
    for D in 2:32
        for mode in modes

            # Calculating nthreads and nblocks
            if mode === Val(:conseq)
                nthreads = max(256, 1 << (ceil(Int, log2(D^2))))
                n_mats_per_block = nthreads ÷ (D * D)
                nblocks = cld(N, n_mats_per_block)
            elseif mode === Val(:indep)
                n_cols_per_warp = max(1, prevpow(2, 32 ÷ D))
                n_elems_per_mat = D ÷ n_cols_per_warp * 32 + (D % n_cols_per_warp) * D
                n_mats_per_block = 1
                nthreads = ((n_mats_per_block * n_elems_per_mat + 31) ÷ 32) * 32
                nblocks = cld(N, n_mats_per_block)
            end

            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D, D, N)
            Bs = CUDA.rand(Float32, D, D, N)
            As_cpu = Array(As)
            Bs_cpu = Array(Bs)

            for (A_adj, B_adj) in test_cases
                Cs = CUDA.zeros(Float32, D, D, N)

                CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_matmul!(
                    Cs,
                    As,
                    Bs,
                    Val(A_adj),
                    Val(B_adj),
                    Val(Int32(D)),
                    Val(Int32(nthreads)),
                    Int32(N),
                    Val(:large),
                    mode,
                )

                # CPU comparison
                Cs_cpu = similar(As_cpu)
                for i in 1:N
                    if A_adj && B_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i]' * Bs_cpu[:, :, i]'
                    elseif A_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i]' * Bs_cpu[:, :, i]
                    elseif B_adj
                        Cs_cpu[:, :, i] = As_cpu[:, :, i] * Bs_cpu[:, :, i]'
                    else
                        Cs_cpu[:, :, i] = As_cpu[:, :, i] * Bs_cpu[:, :, i]
                    end
                end

                max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
                @test max_error < 1e-5
            end
        end
    end
end