@testitem "Upper Triangular Backward Solve (out-of-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create upper triangular matrices
            Us = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D, D)
                Us[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D, D, N)
            Cs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Cs, Us, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
            )

            # CPU comparison
            Us_cpu = Array(Us)
            Bs_cpu = Array(Bs)
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Lower Triangular Forward Solve (out-of-place)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            # Create lower triangular matrices
            Ls = CUDA.zeros(Float32, D, D, N)
            for i in 1:N
                L_temp = CUDA.rand(Float32, D, D)
                Ls[:, :, i] = LowerTriangular(L_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D, D, N)
            Cs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Cs, Ls, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
            )

            # CPU comparison
            Ls_cpu = Array(Ls)
            Bs_cpu = Array(Bs)
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = LowerTriangular(Ls_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end