@testitem "Matrix Multiplication" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels

    include("matmul_kernels.jl")

    # Test parameters
    N = 2^12 + 113
    nthreads = 2^8

    # Test for both independent and consequtive modes
    modes = (Val(:indep), Val(:conseq))

    # Test all four combinations
    test_cases = [(false, false), (true, false), (false, true), (true, true)]

    # Accuracy tests
    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
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
                    Val(:small),
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

@testitem "Matrix Multiplication (non_square)" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels

    include("matmul_kernels.jl")

    # Test parameters
    N = 2^12 + 113
    nthreads = 2^8

    # Accuracy tests
    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            As_cpu = rand(Float32, D1, D2, N)
            Bs_cpu = rand(Float32, D2, D1, N)

            As = cu(As_cpu)
            Bs = cu(Bs_cpu)

            Cs = CUDA.zeros(Float32, D1, D1, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_matmul!(
                Cs,
                As,
                Bs,
                Val(Int32(D1)),
                Val(Int32(D2)),
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
            )
            Cs_result = Array(Cs)

            # CPU comparison
            Cs_cpu = zeros(Float32, D1, D1, N)
            for i in 1:N
                Cs_cpu[:, :, i] = As_cpu[:, :, i] * Bs_cpu[:, :, i]
            end

            # Cs_cpu_resized = zeros(Float32, D, D, N)
            # Cs_cpu_resized[1:D1, 1:D1, :] .= Cs_cpu

            max_error = maximum(abs.(Cs_result .- Cs_cpu))
            @test max_error < 1e-5
        end
    end
end

@testitem "Matrix Multiplication Throughput (indep)" begin
    using PerformanceTestTools

    PerformanceTestTools.@include("throughput_script.jl")
end