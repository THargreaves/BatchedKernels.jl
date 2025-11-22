@testitem "Kalman (small)" begin
    using CUDA
    using CUDA: i32
    using LinearAlgebra
    using BatchedKernels
    import StaticArrays: @MVector

    # Test parameters
    nthreads = 2^8
    N = 2^20 + 113

    # Test for both independent and consequtive modes
    # modes = (Val(:indep), Val(:conseq))
    modes = (Val(:indep),)

    function cpu_kalman_cov(P, A, Q, H, R)
        # Predict step
        P_pred = A * P * A' + Q
        # Update step
        S = H * P_pred * H' + R
        K = P_pred * H' / S
        P_new = P_pred - K * S * K'
        return P_new, P_pred, K
    end

    for D in 2:14
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        # Generate test data
        A_elem = rand(Float32, D, D) / Float32(D)
        Q_elem = rand(Float32, D, D) / Float32(D)^2
        Q_elem = Q_elem * Q_elem' + 0.01f0 * I

        H_elem = rand(Float32, D, D) / Float32(D)
        R_elem = rand(Float32, D, D) / Float32(D)^2
        R_elem = R_elem * R_elem' + 0.01f0 * I

        P_cpu = Array{Float32}(undef, D, D, N)
        for i in 1:N
            P_i = rand(Float32, D, D) / Float32(D)
            P_i = P_i * P_i' + 0.1f0 * I
            P_cpu[:, :, i] = P_i
        end

        P_in = CuArray(P_cpu)

        A_gpu = CuArray(A_elem)
        Q_gpu = CuArray(Q_elem)
        H_gpu = CuArray(H_elem)
        R_gpu = CuArray(R_elem)

        for mode in modes
            P_out = CuArray{Float32}(undef, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_kalman!(
                P_out,
                P_in,
                A_gpu,
                Q_gpu,
                H_gpu,
                R_gpu,
                Val(Int32(D)),
                Val(Int32(nthreads)),
                Int32(N),
                Val(:small),
                mode,
            )

            # Validate P_new (complete Kalman filter output)
            P_out_cpu = Array(P_out)
            max_error_P = 0.0
            n_tested = min(10000, N)

            for i in 1:n_tested
                P_new_ref, P_pred_ref, K_ref = cpu_kalman_cov(
                    P_cpu[:, :, i], A_elem, Q_elem, H_elem, R_elem
                )
                P_new_gpu = P_out_cpu[:, :, i]
                error = maximum(abs.(LowerTriangular(P_new_ref) - LowerTriangular(P_new_gpu)))
                max_error_P = max(max_error_P, error)
            end

            @test max_error_P < 1e-5
        end
    end
end

@testitem "Matrix Multiplication (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

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

@testitem "Matrix Multiplication (large)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

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

@testitem "Matrix Subtraction (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    function kernel_sub!(Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{mode}) where {D,nthreads,mode}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

        shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

        # Load B
        intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N, Val(:small))
        
        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            # Create dual-access matrices
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
            B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))
            C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id, Val(:small))

            # Perform operation
            batch_op!(-, C, A, B, d, Val(D), Val(:small))
        end

        # Store C
        dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Cs, shmem_1, Val(D), Val(nthreads), N, Val(:small), Val(mode))

        return nothing
    end

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D, D, N)
            Bs = CUDA.rand(Float32, D, D, N)
            Cs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_sub!(
                Cs, As, Bs, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode,
            )

            # CPU comparison
            As_cpu = Array(As)
            Bs_cpu = Array(Bs)
            Cs_cpu = As_cpu .- Bs_cpu

            max_error = maximum(abs.(Array(Cs) .- Cs_cpu))
            @test max_error < 1e-5
        end
    end
end

@testitem "Cholesky Decomposition (in-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

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

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_cholesky_inplace!(
                Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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
                Us, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), Val(:small), mode,
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

@testitem "Upper Triangular Backward Solve (out-of-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

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

@testitem "Lower Triangular Forward Solve (out-of-place) (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

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

@testitem "Transpose (small)" begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    # Test both modes
    modes = (Val(:indep), Val(:conseq))

    function kernel_transpose!(
        Bs, As, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{mode},
    ) where {D,nthreads,mode}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        n_mats_per_block = n_warps * n_mats_per_warp
        padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        bid = blockIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)
        grid_mtrx_id = warp_matrix_id + (bid - 1i32) * n_mats_per_block

        shmem_elems = (n_mats_per_warp * D + padding) * D * n_warps
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load A
        intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N, Val(:small))
        interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N, Val(:small))

        if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
            # Create dual-access matrices
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id, Val(:small))
            B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id, Val(:small))

            # Perform transpose
            batch_op!(transpose, B, A, d, Val(D), Val(:small))
        end

        # Store B
        dual_to_interm_transfer!(shmem_3, shmem_2, Val(D), Val(nthreads), N, Val(:small))
        intermediate_layout_write!(Bs, shmem_3, Val(D), Val(nthreads), N, Val(:small), Val(mode))

        return nothing
    end

    for D in 2:15
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        for mode in modes
            CUDA.seed!(1234)

            As = CUDA.rand(Float32, D, D, N)
            Bs = CUDA.zeros(Float32, D, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_transpose!(
                Bs, As, Val(Int32(D)), Val(Int32(nthreads)), Int32(N), mode,
            )

            # CPU comparison
            As_cpu = Array(As)
            Bs_cpu = permutedims(As_cpu, (2, 1, 3))

            max_error = maximum(abs.(Array(Bs) .- Bs_cpu))
            @test max_error < 1e-5
        end
    end
end
