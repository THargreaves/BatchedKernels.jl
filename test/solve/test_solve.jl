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

@testitem "Upper Triangular Backward Solve (out-of-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Us = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Us[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Cs = CUDA.zeros(Float32, D1, D2, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Cs, Us, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
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

@testitem "Upper Triangular Backward Solve (in-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Us = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Us[:, :, i] = UpperTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Us_cpu = Array(Us)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Bs, Us, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Bs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Upper Triangular Backward Solve (in-place, trig)" begin
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
            Us_cpu = Array(Us)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_backward_solve!(
                Us, Us, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = UpperTriangular(Us_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Us) .- Cs_cpu))
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

@testitem "Lower Triangular Forward Solve (out-of-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Ls = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Ls[:, :, i] = LowerTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Cs = CUDA.zeros(Float32, D1, D2, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Cs, Ls, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
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

@testitem "Lower Triangular Forward Solve (in-place, non-square)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    include("solve_kernels.jl")

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    for D1 in 2:15
        for D2 in 2:15
            D = max(D1, D2)
            nblocks = cld(N, nthreads//32 * (32 ÷ D))

            CUDA.seed!(1234)

            # Create upper triangular matrices
            Ls = CUDA.zeros(Float32, D1, D1, N)
            for i in 1:N
                U_temp = CUDA.rand(Float32, D1, D1)
                Ls[:, :, i] = LowerTriangular(U_temp) + 0.5f0 * I  # ensure well-conditioned
            end
            Bs = CUDA.rand(Float32, D1, D2, N)
            Ls_cpu = Array(Ls)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Bs, Ls, Bs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N), Val(:small),
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = LowerTriangular(Ls_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Bs) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Lower Triangular Forward Solve (in-place, triangular)" begin
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
            Ls_cpu = Array(Ls)
            Bs_cpu = Array(Bs)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_forward_solve!(
                Ls, Ls, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
            )

            # CPU comparison
            Cs_cpu = similar(Bs_cpu)
            for i in 1:N
                Cs_cpu[:, :, i] = LowerTriangular(Ls_cpu[:, :, i]) \ Bs_cpu[:, :, i]
            end

            max_error = maximum(abs.(Array(Ls) .- Cs_cpu))
            @test max_error < 1e-3
        end
    end
end

@testitem "Symmetric solve (vmap)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12 + 113

    function solve(A, B, dummy)
        return B / Symmetric(A)
    end

    for D1 in 2:13
        for D2 in 2:13
            Dmax = max(D1, D2)
            for extra in 0:2
                D = Dmax + extra
                dummy = CUDA.zeros(Float32, D, D, N)

                CUDA.seed!(1234)

                As = CUDA.zeros(Float32, D1, D1, N)
                for i in 1:N
                    A_temp = CUDA.rand(Float32, D1, D1)
                    As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
                end

                Bs = CUDA.rand(Float32, D2, D1, N)

                As_cpu = Array(As)
                Bs_cpu = Array(Bs)

                solve_vmap = BatchedKernels.vmap(solve)
                Cs = solve_vmap(As, Bs, dummy)
                Cs_result = Array(Cs)

                Cs_cpu = similar(Bs_cpu)
                for i in 1:N
                    Cs_cpu[:, :, i] = solve(As_cpu[:, :, i], Bs_cpu[:, :, i], 0)
                end

                max_error = maximum(abs.(Cs_result .- Cs_cpu))

                @test max_error < 1e-3
            end
        end
    end
end

@testitem "Symmetric solve (shared, vmap)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    N = 2^12 + 113

    function solve(A, B, dummy)
        return B / Symmetric(A)
    end

    for D1 in 2:13
        for D2 in 2:13
            Dmax = max(D1, D2)
            for extra in 0:1
                D = Dmax + extra
                dummy = CUDA.zeros(Float32, D, D, N)

                CUDA.seed!(1234)

                # Make A shared
                A_temp = CUDA.rand(Float32, D1, D1)
                A = A_temp * A_temp' + 0.1f0 * I

                Bs = CUDA.rand(Float32, D2, D1, N)

                A_cpu = Array(A)
                Bs_cpu = Array(Bs)

                solve_vmap = BatchedKernels.vmap(
                    solve,
                    in_type = (:shared, :batched, :batched)
                )
                Cs = solve_vmap(A, Bs, dummy)
                Cs_result = Array(Cs)

                Cs_cpu = similar(Bs_cpu)
                for i in 1:N
                    Cs_cpu[:, :, i] = solve(A_cpu, Bs_cpu[:, :, i], 0)
                end

                max_error = maximum(abs.(Cs_result .- Cs_cpu))

                @test max_error < 1e-3


                # Make B shared
                CUDA.seed!(1234)

                As = CUDA.zeros(Float32, D1, D1, N)
                for i in 1:N
                    A_temp = CUDA.rand(Float32, D1, D1)
                    As[:, :, i] = A_temp * A_temp' + 0.1f0 * I
                end

                B = CUDA.rand(Float32, D2, D1)

                As_cpu = Array(As)
                B_cpu = Array(B)

                solve_vmap = BatchedKernels.vmap(
                    solve,
                    in_type = (:batched, :shared, :batched)
                )
                Cs = solve_vmap(As, B, dummy)
                Cs_result = Array(Cs)

                Cs_cpu = zeros(Float32, D2, D1, N)
                for i in 1:N
                    Cs_cpu[:, :, i] = solve(As_cpu[:, :, i], B_cpu, 0)
                end

                max_error = maximum(abs.(Cs_result .- Cs_cpu))

                @test max_error < 1e-3
            end
        end
    end
end